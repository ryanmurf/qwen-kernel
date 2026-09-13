import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import audit_native_flash_campaign as audit

HERE = Path(__file__).resolve().parent


class CampaignAuditTest(unittest.TestCase):
    def test_recorded_combined_and_profile(self):
        gate = HERE/'results-halo-combined-gate.jsonl'
        profile = HERE/'results-halo-long-profile.jsonl'
        if not gate.exists() or not profile.exists():
            self.skipTest('completed campaign fixtures unavailable')
        self.assertEqual(audit.validate_combined(audit.read(gate),audit.read(
            HERE/'results-halo-last-head-full-gate.jsonl'))['result'],'PASS')
        self.assertEqual(audit.validate_profile(audit.read(profile),
            (HERE/'results-halo-long-profile.log').read_text())['steps'],100)

    def combined_fixture(self):
        reference = audit.read(HERE/'results-halo-last-head-full-gate.jsonl')
        rows = copy.deepcopy(reference)
        meta = rows[0]
        meta.update(experiment='native-combined-gate-v1',
                    identity={'library':meta['library_sha256'],'model':meta['model'],'shaders':meta['shaders']})
        meta['environment']['QK_FLASH_ATTN_BATCH']='vec4'
        hashes = {r['prompt_rows']:r['logit_sha256'] for r in reference if r['type']=='case_pass'}
        for row in rows:
            if row['type']=='row':
                row['logit_sha256']=hashes[row['prompt_rows']][row['row']]
            if row['type']=='result':
                row.clear()
                row.update(type='result',result='PASS',cases=15,compared_rows=664,
                           bit_exact=True,identities_unchanged=True,full_model=True)
        return rows,reference

    def test_combined_coverage_and_corruption(self):
        rows,reference = self.combined_fixture()
        self.assertEqual(audit.validate_combined(rows,reference)['compared_full_vocabulary_rows'],664)
        corruptions = [lambda x:x.pop(-3),
                       lambda x:x[0]['environment'].update(QK_FLASH_COOPMAT='1'),
                       lambda x:x[0]['identity'].update(library='0'*64),
                       lambda x:next(r for r in x if r['type']=='row').update(logit_sha256='0'*64),
                       lambda x:next(r for r in x if r['type']=='row').update(seconds=float('inf')),
                       lambda x:x.pop()]
        for mutate in corruptions:
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(rows)
                mutate(changed)
                with self.assertRaises((ValueError,KeyError)):
                    audit.validate_combined(changed,reference)

    def profile_fixture(self):
        rows = [dict(type='metadata',experiment='native-long-profile-v1',context=32768,chunk=512,
                     environment={**audit.full.ENV,'QK_FLASH_PROFILE':'1','QK_FLASH_REPLAY':'0',
                                  'QK_FLASH_TIMING':'1','QK_SHADER_DIR':'/frozen/shaders'})]
        log = []
        for size in (2048,8192,16384):
            for phase,index,base,width in ([('prefill',i,b,512) for i,b in enumerate(range(0,size,512))]+
                                           [('decode',i,size+i,1) for i in range(16)]):
                marker = dict(phase=phase,prompt_tokens=size,index=index,base=base,width=width)
                rows.append(dict(type='step',**marker,seconds=.01,greedy=16))
                label = f'batched forward, {width} tokens' if width>1 else 'serial forward, 1 token'
                log.extend(['[campaign begin] '+json.dumps(marker),
                            f'[flash profile] {label}: 4.000 ms GPU',
                            '[flash profile] fa_attn_srv.spv 3.000 ms 75.0% (12 dispatches, 250.0 us each)',
                            '[flash profile] gemv_q5_k.spv 1.000 ms 25.0% (48 dispatches, 20.8 us each)',
                            '[campaign end] '+json.dumps(marker)])
            rows.append(dict(type='case_pass',prompt_tokens=size))
        rows.extend([dict(type='result',result='PASS',cases=3),dict(type='closed')])
        return rows,'\n'.join(log)

    def test_profile_coverage_and_corruption(self):
        rows,log = self.profile_fixture()
        result = audit.validate_profile(rows,log)
        self.assertEqual(result['steps'],100)
        self.assertEqual(result['summary'][-1]['category_percent']['attention'],75)
        for broken in (log.replace('[campaign end]','[missing end]',1),
                       log.replace('4.000 ms GPU','9.000 ms GPU',1),
                       log.replace('512 tokens','511 tokens',1),
                       log.replace('3.000 ms 75.0%','9.000 ms 75.0%',1)):
            with self.assertRaises(ValueError):
                audit.validate_profile(rows,broken)
        rows.pop(1)
        with self.assertRaises(ValueError):
            audit.validate_profile(rows,log)

    def test_resource_audit_and_corruption(self):
        private = Path('/home/ryan/qk-last-head-full-VjRrMt')
        if not private.exists():
            self.skipTest('private resource fixture unavailable')
        controller = json.loads((private/'http-last.controller.json').read_text())
        rows = audit.read(private/'http-last.observer.jsonl')
        self.assertEqual(audit.validate_resources(controller,rows,True)['result'],'PASS')
        for change in ('gap','external','low_memory','event','pid'):
            bad = copy.deepcopy(rows)
            live = next(r for r in bad if r['devices'])
            if change=='gap':
                bad[2]['elapsed']+=30
            elif change=='external':
                next(d for d in live['devices'].values() if d['drm-pdev']!='0000:c1:00.0')['drm-engine-compute']='1 ns'
            elif change=='low_memory':
                live['memory']['MemAvailable']=1
            elif change=='event':
                live['events']['oom_kill']='1'
            else:
                live['properties']['MainPID']='9999999'
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_resources(controller,bad,True)

    def test_http_matrix_and_corruption(self):
        # Synthetic campaign assembled from old recorded fixture rows. These
        # are auditor unit-test inputs, never published benchmark measurements.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='qk-campaign-audit-') as name:
            folder = Path(name)
            def write(path,value,lines=False):
                with Path(path).open('w') as stream:
                    if lines:
                        for row in value:
                            stream.write(json.dumps(row)+'\n')
                    else:
                        json.dump(value,stream)
            write(folder/'http-build.json',{})
            gate,reference = self.combined_fixture()
            write(folder/'gate.jsonl',gate,True)
            previous = audit.read(HERE/'results-halo-last-head-http-last.jsonl')
            original_read = audit.read
            for index,mode in enumerate(audit.ORDER):
                prefix = folder/f'http-{index:02d}-{mode}'
                env = {**audit.full.ENV,'QK_SHADER_DIR':'/home/ryan/qk-last-head-IIfPmw/build/shaders',
                       'QK_REASONING_EFFORT':'xhigh','QK_PREFILL_CHUNK':'512',
                       'QK_FLASH_PREFILL_LAST':'1' if mode in ('last','combined') else '0',
                       'QK_FLASH_ATTN_BATCH':'vec4' if mode in ('vec4','combined') else 'baseline'}
                command = ['/home/ryan/qk-last-head-IIfPmw/rust-release/release/server',
                    '--model','/home/ryan/models/Qwen3.8-Flash-Next-Uncensored-Q5_K_M/Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf',
                    '--engine-lib','/home/ryan/qk-last-head-IIfPmw/build/libqk.so',
                    '--host','127.0.0.1','--port','8194','--slots','1','--ctx','32768',
                    '--chunk','1','--queue','4','--local-driver','--chat-template','auto']
                ctrl = dict(launch_index=index,mode=mode,order=audit.ORDER,actual_environment=env,
                    actual_command=command,server_command=command,default_promotion=False,
                    build_sha256=audit.sha(folder/'http-build.json'),full_gate_sha256=audit.sha(folder/'gate.jsonl'),
                    sampled_token_sha256='1'*64)
                write(str(prefix)+'.controller.json',ctrl)
                write(str(prefix)+'.observer.jsonl',[],True)
                meta = dict(experiment='native-combined-http-v1',mode=mode,launch_index=index,
                    order=audit.ORDER,build={},actual_environment=env,actual_command=command,server_command=command,
                    context=32768,precision='native F32',mtp=False,default_promotion=False,slots=1,prefill_chunk=512,
                    http_returncode=0,sampled_token_sha256='1'*64)
                for group,sizes,count in [('short',[128,512,2048,8192],128),('long',[16384],512)]:
                    header = copy.deepcopy(previous[0])
                    header.update(sizes=sizes,repetitions=1,requested_output_tokens=count,metadata=meta,
                                  config_sha256=audit.pm.digest(meta))
                    cells = [copy.deepcopy(r) for r in previous if r['type'] in ('prefill','decode') and
                             r['repetition']==1 and r['prompt_tokens'] in sizes]
                    for cell in cells:
                        cell.update(requested_output_tokens=count,config_sha256=header['config_sha256'])
                        if cell['type']=='decode':
                            cell['streamed_tokens']=count
                    write(str(prefix)+f'.{group}.jsonl',[header,*cells,dict(type='run_complete')],True)
                code = [dict(type='metadata',experiment='native-code-stream-v1',max_output_tokens=256,
                        generated_code_executed=False,workload_sha256='5'*64),
                        *[dict(type='code',name=n,complete_chunk_token_counts=True,
                        streamed_tokens=256,prompt_sha256='2'*64,output_token_sha256='3'*64,output_text_sha256='4'*64,
                        requested_output_tokens=256,prompt_tokens=512,server_prompt_tokens=512,
                        ttft_seconds=3,decode_tokens_per_second=30,stream_total_seconds=12)
                        for n in ('python-lru-review','vulkan-attention-review')],dict(type='complete')]
                write(str(prefix)+'.code.jsonl',code,True)
            # Resource corruption is tested independently above; isolate HTTP
            # coverage here so these synthetic records need no fabricated telemetry.
            with patch.object(audit,'validate_resources',return_value={'result':'PASS'}), \
                 patch.object(audit,'read',side_effect=lambda p: reference if str(p)==
                     '/home/ryan/qk-last-head-full-VjRrMt/gate.jsonl' else original_read(p)):
                self.assertEqual(audit.validate_http(folder)['launches'],8)
                path = folder/'http-03-combined.long.jsonl'
                saved = original_read(path)
                mutations = [lambda r:r.pop(1),
                             lambda r:r[0]['metadata'].update(precision='F16'),
                             lambda r:r[2].update(streamed_tokens=128),
                             lambda r:r[2].update(output_token_sha256='0'*64),
                             lambda r:r[2].update(ttft_seconds=float('inf')),
                             lambda r:r[2].update(prompt_sha256='0'*64),
                             lambda r:r[2].update(server_prompt_tokens=1),
                             lambda r:r[2].update(config_sha256='0'*64)]
                for mutate in mutations:
                    bad = copy.deepcopy(saved)
                    mutate(bad)
                    write(path,bad,True)
                    with self.subTest(mutate=mutate),self.assertRaises((ValueError,KeyError)):
                        audit.validate_http(folder)
                write(path,saved,True)


if __name__=='__main__':
    unittest.main()
