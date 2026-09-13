"""Replay completed evidence and reject missing/changed measurements."""
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

from audit_native_decode_loads import validate_resources, compare

DATA=Path(__file__).parent/'results-halo-expert-prep'
def rows(path): return [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]

def validate_cells(data,modes):
    expected=[(l,a,n,scale,m,i) for l in (0,47)
              for a,n,scale in ((32,1,1),(32,8,1),(32,16,1),(32,17,1),(1,64,8),(16,3,.25))
              for i,m in enumerate((modes[0],modes[1],modes[2],modes[2],modes[1],modes[0]))]
    if data[-1]!=dict(type='result',result='PASS',cells=72,full_model=False): raise ValueError('incomplete')
    cells=[r for r in data if r['type']=='cell']
    if [(r['layer'],r['active_experts'],r['pairs_per_expert'],r['q_scale'],r['mode'],r['order'])
        for r in cells]!=expected: raise ValueError('coverage/order')
    for r in cells:
        if r['result']!='PASS' or not math.isfinite(r['gpu_us']) or r['gpu_us']<=0:
            raise ValueError('timing/status')
        if any(r[k]!=0 for k in ('bit_mismatches','nonfinite','padding_changes','fp64_misses')):
            raise ValueError('numerical failure')
    return len(cells)

class ExpertPrepEvidence(unittest.TestCase):
    def test_expert_coverage_and_resources(self):
        for name,modes in (('layout',('baseline','words','soa')),('split',('baseline','soa','split'))):
            raw=DATA/f'{name}-raw.log'
            record=json.loads((DATA/f'{name}-result.json').read_text())
            self.assertEqual(record['result'],'PASS')
            self.assertTrue(record['restored'])
            self.assertFalse(record['default_promotion'])
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),record['log_sha256'])
            self.assertEqual(validate_cells(rows(raw),modes),72)
            telemetry=rows(DATA/f'{name}-observer.jsonl')
            got=validate_resources(telemetry,telemetry[0]['unit'],record['final_unit'],False)
            self.assertEqual(got,record['resources'])
            for port in (8194,8091,8092):
                checks=rows(DATA/f'{name}-restored-http{port}.jsonl')
                self.assertEqual(len(checks),9 if port==8194 else 8 if port==8091 else 2)
                for r in checks:
                    if r['test']=='stream_benchmark':
                        self.assertTrue(r['coherent_counting_prefix'])
                        self.assertEqual(r['streamed_tokens'],96)
                    else: self.assertEqual(r['result'],'PASS')

    def test_missing_or_corrupted_expert_cells_rejected(self):
        original=rows(DATA/'split-raw.log')
        for key,value in (('bit_mismatches',1),('gpu_us',float('nan')),('mode','baseline')):
            bad=copy.deepcopy(original)
            next(r for r in bad if r['type']=='cell' and r['mode']=='split')[key]=value
            with self.assertRaises(ValueError): validate_cells(bad,('baseline','soa','split'))
        with self.assertRaises(ValueError): validate_cells(original[:-1],('baseline','soa','split'))

    def test_cpu_evidence_and_unchanged_serving(self):
        record=json.loads((DATA/'cpu-result.json').read_text())
        data=rows(DATA/'cpu-raw.jsonl')
        self.assertEqual(record['result'],'PASS')
        self.assertFalse(record['gpu_used'])
        self.assertFalse(record['production_changed'])
        self.assertEqual(record['serving_before'],record['serving_after'])
        self.assertEqual(record['final_unit']['SubState'],'exited')
        self.assertEqual(record['final_unit']['MemorySwapPeak'],'0')
        self.assertLessEqual(int(record['final_unit']['MemoryPeak']),2**30)
        self.assertEqual(hashlib.sha256((DATA/'cpu-raw.jsonl').read_bytes()).hexdigest(),record['raw_sha256'])
        for n in (128,512,2048):
            measured=[r for r in data if r['type']=='prep' and r['tokens']==n]
            self.assertEqual([r['mode'] for r in measured],['portable','avx512','avx512','portable']*2)
            self.assertEqual(len({r['output_fnv64'] for r in measured}),1)
            self.assertTrue(all(r['result']=='PASS' and r['ms']>0 and r['repeats']==30 for r in measured))

    def test_failed_decode_campaign_not_promoted(self):
        root=DATA.parent/'results-halo-decode-loads'
        decision=json.loads((root/'decision.json').read_text())
        result=json.loads((root/'result.json').read_text())
        self.assertFalse(decision['loads_promoted'])
        self.assertTrue(result['error'])
        self.assertTrue(result['restored'])
        incomplete=[rows(root/f'http-{i}-{mode}.matrix.jsonl') for i,mode in enumerate(('serial','loads','loads'))]
        with self.assertRaises(ValueError): compare(incomplete)

if __name__=='__main__': unittest.main()
