import unittest

from audit_native_last_head_full import SHORT, validate_gate, validate_http
from native_last_head_full import ENV, row_plan


def gate_fixture():
    rows = [{'type':'metadata','experiment':'native-last-head-full-v1','context':32768,'chunk':512,
             'admission':{'admitted':True},'library_sha256':'a'*64,'fixture_sha256':'b'*64,
             'environment':{**ENV,'QK_SHADER_DIR':'/test/shaders'}}]
    for size in SHORT + [8501,16693]:
        plan = row_plan(size,size not in SHORT)
        for iteration,last in enumerate((False,True,True,False)):
            for index,(base,width) in enumerate(plan):
                rows.append(dict(type='row',prompt_rows=size,iteration=iteration,last_only=last,
                                 row=index,base=base,width=width,greedy=42,seconds=1.0))
            rows.append(dict(type='case_pass',prompt_rows=size,iteration=iteration,last_only=last,
                             compared_rows=len(plan),logit_sha256=['c'*64]*len(plan)))
    return rows + [dict(type='result',result='PASS',cases=15,compared_rows=498,
                        bit_exact=True,identities_unchanged=True),dict(type='closed')]


def http_fixture(mode):
    common=dict(run_id=mode,backend=mode,model_id='test-model',fixture_sha256='a'*64,context=32768,
                config_sha256=mode,requested_output_tokens=128)
    metadata=dict(mode=mode,http_returncode=0,actual_environment={**ENV,'QK_SHADER_DIR':'/test/shaders',
                  'QK_PREFILL_CHUNK':'512','QK_REASONING_EFFORT':'xhigh',
                  'QK_FLASH_PREFILL_LAST':'1' if mode=='last' else '0'},
                  build={},full_gate_sha256='b'*64,sampled_token_sha256='c'*64,
                  precision='native F32',mtp=False,slots=1,prefill_chunk=512,
                  experiment='native-last-head-http-v1',context=32768,default_promotion=False)
    sizes=[128,512,2048,8192,16384]
    rows=[dict(type='run_start',**common,sizes=sizes,repetitions=3,metadata=metadata)]
    for size in sizes:
        for repetition in (1,2,3):
            for kind in ('prefill','decode'):
                rows.append(dict(type=kind,**common,prompt_tokens=size,repetition=repetition,
                    server_prompt_tokens=size,prompt_sha256='d'*64,output_tokens=[16],prefill_plus_one_token_seconds=1.,
                    exact_output_length=True,complete_chunk_token_counts=True,coherent_counting_prefix=True,
                    streamed_tokens=128,ttft_seconds=1.,decode_tokens_per_second=32.,stream_total_seconds=5.1,
                    output_token_sha256='e'*64,output_text_sha256='f'*64))
    return rows+[dict(type='run_complete',**common)]


class FullLastHeadTest(unittest.TestCase):
    def test_plans_are_contiguous(self):
        for n,chunked,expected_rows,tail in [(1,False,4,66),(513,False,4,66),(8501,True,49,32),(16693,True,65,32)]:
            plan=row_plan(n,chunked)
            self.assertEqual(len(plan),expected_rows)
            self.assertEqual(plan[0][0],0)
            self.assertTrue(all(a+w==b for (a,w),(b,_) in zip(plan,plan[1:])))
            self.assertEqual(sum(w for _,w in plan),n+tail)
            if chunked:
                self.assertEqual([w for _,w in plan if w not in (1,512)],[309])

    def test_complete_gate(self):
        self.assertEqual(validate_gate(gate_fixture())['compared_logit_rows'],498)

    def test_gate_rejects_corruption(self):
        for mutation in ('missing_row','missing_case','hash','base','flag','result','unclosed','greedy','environment','infinite_time'):
            with self.subTest(mutation=mutation):
                rows=gate_fixture()
                if mutation=='missing_row': rows.pop(1)
                elif mutation=='missing_case': rows.pop(next(i for i,r in enumerate(rows) if r['type']=='case_pass'))
                elif mutation=='hash': next(r for r in rows if r['type']=='case_pass' and r['iteration']==1)['logit_sha256'][0]='0'*64
                elif mutation=='base': rows[1]['base']=1
                elif mutation=='flag': rows[1]['last_only']=True
                elif mutation=='result': rows[-2]['bit_exact']=False
                elif mutation=='unclosed': rows.pop()
                elif mutation=='greedy': next(r for r in rows if r['type']=='row' and r['iteration']==1)['greedy']=43
                elif mutation=='environment': rows[0]['environment']['QK_FLASH_COOPMAT']='1'
                elif mutation=='infinite_time': rows[1]['seconds']=float('inf')
                with self.assertRaises(ValueError): validate_gate(rows)

    def test_complete_http_pair(self):
        result=validate_http(http_fixture('all'),http_fixture('last'))
        self.assertEqual(result['paired_decode_outputs'],15)

    def test_http_rejects_corruption(self):
        for mutation in ('missing','prompt','output','sampling','build','environment','flag','count','tokenizer','cache',
                         'both_environment','both_precision','both_mtp','both_chunk','bad_sample_hash','infinite_time'):
            with self.subTest(mutation=mutation):
                a,b=http_fixture('all'),http_fixture('last')
                if mutation=='missing': b.pop(1)
                elif mutation=='prompt': b[1]['prompt_sha256']='0'*64
                elif mutation=='output': b[2]['output_token_sha256']='0'*64
                elif mutation=='sampling': b[0]['metadata']['sampled_token_sha256']='0'*64
                elif mutation=='build': b[0]['metadata']['build']={'different':True}
                elif mutation=='environment': b[0]['metadata']['actual_environment']['QK_FLASH_COOPMAT']='1'
                elif mutation=='flag': b[0]['metadata']['actual_environment']['QK_FLASH_PREFILL_LAST']='0'
                elif mutation=='count': b[2]['streamed_tokens']=127
                elif mutation=='tokenizer': b[1]['fixture_sha256']='0'*64
                elif mutation=='cache': b[1]['server_prompt_tokens']=1
                elif mutation=='both_environment':
                    for rows in (a,b): rows[0]['metadata']['actual_environment']['QK_FLASH_COOPMAT']='1'
                elif mutation=='both_precision':
                    for rows in (a,b): rows[0]['metadata']['precision']='F16'
                elif mutation=='both_mtp':
                    for rows in (a,b): rows[0]['metadata']['mtp']=True
                elif mutation=='both_chunk':
                    for rows in (a,b): rows[0]['metadata']['prefill_chunk']=128
                elif mutation=='bad_sample_hash':
                    for rows in (a,b): rows[0]['metadata']['sampled_token_sha256']='bad'
                elif mutation=='infinite_time': b[2]['ttft_seconds']=float('inf')
                with self.assertRaises(ValueError): validate_http(a,b)


if __name__=='__main__':
    unittest.main()
