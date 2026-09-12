"""CPU-only tests; fake streams only, no GPU/model loads."""
import contextlib
import copy
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import llama_session_matrix as m


class SessionMatrixTests(unittest.TestCase):
    def measure(self, events):
        with patch.object(m.pm, 'post', return_value=contextlib.nullcontext(None)), \
                patch.object(m.pm, 'sse', return_value=iter(events)):
            return m.measure('http://example.invalid', [1]*64, 3, False, 2)

    def test_request_is_greedy_and_explicit(self):
        for reuse in (True, False):
            b=m.body([1,2],128,reuse)
            self.assertEqual(b['cache_prompt'],reuse)
            self.assertEqual(b['id_slot'],0)
            self.assertEqual(b['temperature'],0)
            self.assertNotIn('n_probs',b)
            self.assertNotIn('spec_synth_len',b)

    def test_grouped_events_count_tokens_not_events(self):
        r=self.measure([{'tokens':[16,11],'content':'1,'}, {'tokens':[17],'content':'2'},
                        {'stop':True,'id_slot':0,'timings':{'cache_n':0,'prompt_n':64,'predicted_n':3}}])
        self.assertEqual(r['streamed_tokens'],3)
        self.assertEqual(r['events'],2)
        self.assertEqual(r['first_event_tokens'],2)
        self.assertTrue(r['exact_output_length'])
        self.assertTrue(r['coherent_counting_prefix'])
        r.update(cache_condition='fresh',prompt_tokens=64)
        self.assertFalse(m.audit_cell(r,3))

    def test_missing_final_counts_and_error_fail(self):
        for events in ([{'tokens':[16],'content':'1'}], [{'content':'1'}, {'stop':True}],
                       [{'error':'bad'}], [{'stop':True},{'tokens':[16],'content':'1'}]):
            with self.assertRaises(ValueError): self.measure(events)

    def cell(self, kind, cached):
        return {'type':'measurement','cache_condition':kind,'prompt_tokens':64,
                'exact_output_length':True,'streamed_tokens':3,'output_tokens':[16,11,17],
                'output_token_sha256':m.pm.digest([16,11,17]),'coherent_counting_prefix':True,
                'ttft_seconds':1.,'stream_total_seconds':2.,'decode_tokens_per_second':3.,
                'server_timings':{'cache_n':cached,'prompt_n':64-cached,'predicted_n':3},'server_slot':0}

    def test_cache_hits_use_cache_n_not_retained_length(self):
        row=self.cell('repeat',0); row['server_retained_tokens']=9999
        self.assertFalse(m.audit_cell(row,3))
        self.assertTrue(m.audit_cell(self.cell('repeat',63),3))
        with self.assertRaisesRegex(ValueError,'reused KV'): m.audit_cell(self.cell('fresh',63),3)

    def test_rejects_unknown_or_inconsistent_cache_counts(self):
        for timings in ({'prompt_n':64,'predicted_n':3}, {'cache_n':True,'prompt_n':63,'predicted_n':3},
                        {'cache_n':1,'prompt_n':64,'predicted_n':3}):
            row=self.cell('repeat',0);row['server_timings']=timings
            with self.assertRaises(ValueError):m.audit_cell(row,3)

    def test_rejects_output_and_timing_corruption(self):
        for key,value in [('output_tokens',[99,11,17]),('coherent_counting_prefix',False),
                          ('exact_output_length',False),('server_slot',1),('ttft_seconds',float('nan')),
                          ('decode_tokens_per_second',None)]:
            row=self.cell('fresh',0);row[key]=value
            with self.assertRaises(ValueError):m.audit_cell(row,3)

    def rows(self):
        base={'run_id':'r','backend':'plain','fixture_sha256':'f','model_id':'m','context':1024,
              'config_sha256':m.pm.digest({}),'requested_output_tokens':3}
        head={**base,'type':'run_start','metadata':{},'sizes':[64],'repetitions':1,
              'followup_ids':[1,2],'followup_sha256':m.pm.digest([1,2])}
        rows=[head]
        for kind in m.KINDS:
            row={**base,**self.cell(kind,0 if kind=='fresh' else 63),
                 'base_prompt_tokens':64,'repetition':1,'prompt_sha256':'same'}
            if kind=='followup':
                row['prompt_tokens']=69;row['server_timings']['prompt_n']=6
            rows.append(row)
        rows.append({**base,'type':'run_complete'})
        return rows

    def test_complete_matrix_and_explicit_cache_miss_summary(self):
        rows=self.rows(); summary=m.summarize(rows)
        self.assertTrue(summary['matrix_valid'])
        self.assertEqual([r['actual_cache_hits'] for r in summary['rows']],[0,1,1])
        rows[2]['server_timings'].update(cache_n=0,prompt_n=64)
        self.assertEqual(m.summarize(rows)['rows'][1]['actual_cache_hits'],0)

    def test_partial_reordered_mixed_and_corrupt_header_fail(self):
        rows=self.rows()
        for bad in (rows[:-1], rows[:2]+rows[3:], rows[:1]+list(reversed(rows[1:-1]))+rows[-1:], rows+rows):
            with self.assertRaises(ValueError):m.audit(bad)
        for key,value in [('backend','other'),('config_sha256','bad')]:
            bad=copy.deepcopy(rows);bad[1][key]=value
            with self.assertRaises(ValueError):m.audit(bad)
        bad=copy.deepcopy(rows);bad[0]['followup_ids']=[1,3]
        with self.assertRaises(ValueError):m.audit(bad)

    def test_repeat_and_followup_shape_fail_closed(self):
        for index,key,value in [(2,'prompt_sha256','other'),(3,'prompt_tokens',70)]:
            rows=self.rows();rows[index][key]=value
            with self.assertRaises(ValueError):m.audit(rows)

    def run_fixture(self, directory):
        fixture={'version':1,'model_id':'test','sizes':[64,128],
                 'source_text':'source','suffix_text':'suffix',
                 'source_ids':[10]*128,'suffix_ids':[20,21]}
        fixture['sha256']=m.pm.digest(fixture)
        (directory/'fixture.json').write_text(json.dumps(fixture))
        (directory/'metadata.json').write_text('{}')
        args=argparse.Namespace(fixture=directory/'fixture.json',metadata=directory/'metadata.json',
             output=directory/'result.jsonl',url='http://example.invalid',backend='fake',
             context=1024,sizes=None,repetitions=2,decode_tokens=3,timeout=2,confirm_exclusive=True)
        def tokenize(url, text, special, timeout):
            return {'source':fixture['source_ids'],'suffix':fixture['suffix_ids'],m.FOLLOWUP:[30,31]}[text]
        return args, tokenize

    def test_run_constructs_followup_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,tokenize=self.run_fixture(Path(tmp));calls=[]
            def measure(url,prompt,count,reuse,timeout):
                calls.append((list(prompt),reuse))
                row=self.cell('fresh',0)
                for key in ('type','cache_condition','prompt_tokens'):row.pop(key)
                row['server_timings'].update(cache_n=len(prompt)-1 if reuse else 0,
                                            prompt_n=1 if reuse else len(prompt))
                return row
            with patch.object(m.pm,'tokenize',side_effect=tokenize),patch.object(m,'measure',side_effect=measure), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                m.run(args)
                original=args.output.read_bytes()
                with self.assertRaises(FileExistsError):m.run(args)
                self.assertEqual(args.output.read_bytes(),original)
            rows=[json.loads(s) for s in original.splitlines()]
            self.assertTrue(m.summarize(rows)['matrix_valid'])
            self.assertEqual(len(calls),12)
            for index in range(0,len(calls),3):
                fresh,repeat,followup=calls[index:index+3]
                self.assertEqual(fresh[0],repeat[0])
                self.assertEqual(followup[0],repeat[0]+[16,11,17]+[30,31])
                self.assertEqual([fresh[1],repeat[1],followup[1]],[False,True,True])
                self.assertEqual(rows[index+3]['prompt_sha256'],m.pm.digest(followup[0]))

    def test_run_failure_retains_error_without_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,tokenize=self.run_fixture(Path(tmp))
            with patch.object(m.pm,'tokenize',side_effect=tokenize), \
                    patch.object(m,'measure',side_effect=ValueError('bad stream')), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'bad stream'):m.run(args)
            rows=[json.loads(s) for s in args.output.read_text().splitlines()]
            self.assertEqual([r['type'] for r in rows],['run_start','run_error'])
            with self.assertRaises(ValueError):m.summarize(rows)


if __name__=='__main__':unittest.main()
