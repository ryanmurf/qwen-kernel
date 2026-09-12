#!/usr/bin/env python3
"""CPU-only adversarial tests of the exact prefill-output gate."""
from array import array
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('prefill_gate', Path(__file__).with_name('gpu_qwen4_prefill_gate.py'))
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)

class Runner:
    def __init__(self): self.history = []; self.calls = []; self.resets = 0
    def step(self, tokens, base):
        if base == 0: self.history = []; self.resets += 1
        elif base != len(self.history): raise RuntimeError('noncontiguous feed')
        self.history.extend(tokens); self.calls.append((base,len(tokens)))
        return max(range(8), key=self.row().__getitem__)
    def row(self):
        h = hashlib.sha256(str(self.history).encode()).digest()
        return array('f', [x / 256 for x in h[:8]])

class FlowTests(unittest.TestCase):
    def test_exact_partial_prefix_and_complete_repeat(self):
        runner = Runner(); ids = list(range(1205))
        rows, mf = mod.capture(runner, ids, 181, vocab=8)
        self.assertEqual(len(rows), 1 + 2 + 181)
        self.assertTrue(mf['repeat_exact'] and mf['abi_greedy_exact'])
        self.assertEqual(runner.calls[:3], [(0,1),(0,512),(512,512)])
        self.assertEqual(runner.calls[:184], runner.calls[184:])
        partial = mod.plan(16512,181)
        self.assertEqual([x['width'] for x in partial if x['phase']=='prefill'][-1],459)
        self.assertEqual(len(partial),214)

    def test_partial_final_chunk_output_is_captured(self):
        runner = Runner(); rows, mf = mod.capture(runner,list(range(1042)),17,vocab=8)
        prefill = [e for e in mf['row_plan'] if e['phase']=='prefill']
        self.assertEqual([e['width'] for e in prefill],[512,512,1])
        self.assertEqual(len(rows),21)

    def test_prefill_only_repeat_corruption_is_detected(self):
        class Changed(Runner):
            def row(self):
                row = super().row()
                if self.resets >= 4 and self.calls[-1][1] > 1: row[0] += .01
                return row
        _, mf = mod.capture(Changed(), list(range(1050)),17,vocab=8)
        self.assertFalse(mf['prefill_repeat_exact'])
        self.assertTrue(mf['tail_repeat_exact'] and mf['reset_exact'])
        self.assertFalse(mf['repeat_exact'])

    def test_nonfinite_and_wrong_abi_argmax_rejected(self):
        row = array('f',range(8))
        self.assertEqual(mod.row_properties(row,0,8),(True,False))
        row[0] = math.nan
        self.assertEqual(mod.row_properties(row,7,8),(False,False))
        with self.assertRaises(ValueError): mod.row_properties(row,7,9)
        self.assertFalse(mod.row_properties(array('f',range(8)),True,8)[1])

    def test_bad_workload(self):
        for n,tail in ((0,1),(10,0),(10,10),(1000,513),(10,True)):
            with self.assertRaises(ValueError): mod.plan(n,tail)

    def test_prepare_clears_inherited_knobs(self):
        with patch.dict(os.environ,{'QK_DEVICE':'1','QK_FLASH_TIMING':'1','QK_FLASH_COOPMAT':'1','QK_LAYERS':'1:2'}):
            env = mod.prepare_env('vec4','/shaders')
            self.assertEqual(env['QK_FLASH_ATTN_BATCH'],'vec4')
            self.assertEqual(env['QK_FLASH_GEMM'],'baseline')
            self.assertEqual(env['QK_FLASH_COOPMAT'],'0')
            for key in ('QK_DEVICE','QK_FLASH_TIMING','QK_LAYERS'): self.assertNotIn(key,env)

class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.vocab = patch.object(mod,'VOCAB',8); self.vocab.start()
        self.addCleanup(self.tmp.cleanup); self.addCleanup(self.vocab.stop)
        self.a,self.b = self.root/'baseline.f32',self.root/'vec4.f32'
        self.make(self.a,'baseline'); self.make(self.b,'vec4')

    def make(self,path,policy):
        row_plan = mod.plan(1028,3)
        values = [array('f',[float(j+i) for j in range(8)]).tobytes() for i in range(len(row_plan))]
        payload = b''.join(values); path.write_bytes(payload)
        gemm,attn = mod.POLICIES[policy]
        mf = {'manifest_version':mod.VERSION,'experiment':'native-prefill-exact-v1','policy':policy,
              'out':str(path),'dump_sha256':hashlib.sha256(payload).hexdigest(),'ids':'/ids',
              'ids_sha256':'1'*64,'n':1028,'tail':3,'ctx':2048,'vocab':8,'chunk':512,'row_plan':row_plan,
              'model':{'shards':[{'path':'/model.gguf','size':123,'mtime_ns':1}]},'library_sha256':'2'*64,
              'shaders':{'a.spv':'3'*64},'helper_sha256':mod.helpers(),
              'knobs':{**mod.BASE_ENV,'QK_SHADER_DIR':'/shaders','QK_FLASH_GEMM':gemm,'QK_FLASH_ATTN_BATCH':attn},
              'reset_exact':True,'prefill_repeat_exact':True,'tail_repeat_exact':True,'repeat_exact':True,
              'all_finite':True,'abi_greedy_exact':True,'greedy_ids':[7]*len(row_plan),'seconds':1.,'identities_unchanged':True}
        self.save(path,mf)

    def load(self,path): return json.loads(Path(str(path)+'.json').read_text())
    def save(self,path,mf): Path(str(path)+'.json').write_text(json.dumps(mf))
    def compare(self): return mod.compare_paths(self.a,self.b,'vec4')
    def mutate(self,key,value):
        mf = self.load(self.b); mf[key] = value; self.save(self.b,mf)

    def test_exact_pair_passes_with_prefill_rows(self):
        result = self.compare()
        self.assertEqual(result['result'],'PASS')
        self.assertEqual(result['saved_row_counts'],{'clean':1,'prefill':3,'tail':3})
        self.assertEqual(result['partial_prefill_widths'],[1])
        self.assertFalse(result['default_promotion'])

    def test_prefill_only_corruption_fails_even_with_rebound_hash(self):
        payload = bytearray(self.b.read_bytes()); payload[32:36] = array('f',[.1]).tobytes()
        self.b.write_bytes(payload); self.mutate('dump_sha256',hashlib.sha256(payload).hexdigest())
        result = self.compare()
        self.assertEqual(result['result'],'FAIL')
        self.assertEqual(result['failures'][0]['phase'],'prefill')

    def test_same_nan_payload_cannot_pass(self):
        for path in (self.a,self.b):
            data=bytearray(path.read_bytes()); data[:4]=array('f',[math.nan]).tobytes(); path.write_bytes(data)
            mf=self.load(path); mf['dump_sha256']=hashlib.sha256(data).hexdigest(); self.save(path,mf)
        self.assertEqual(self.compare()['result'],'FAIL')

    def test_missing_keys_in_both_do_not_pass(self):
        for key in mod.REQUIRED:
            with self.subTest(key=key):
                a,b=self.load(self.a),self.load(self.b)
                aa,bb=deepcopy(a),deepcopy(b); del aa[key]; del bb[key]
                self.save(self.a,aa); self.save(self.b,bb)
                with self.assertRaises((ValueError,TypeError,KeyError)): self.compare()
                self.save(self.a,a); self.save(self.b,b)

    def test_identity_and_status_mismatches(self):
        original=self.load(self.b)
        for key,value in (('library_sha256','f'*64),('ids_sha256','e'*64),('tail',4),('ctx',4096),
                          ('shaders',{'a.spv':'a'*64}),('reset_exact',False),('prefill_repeat_exact',False),
                          ('tail_repeat_exact',False),('identities_unchanged',False),('manifest_version',0)):
            with self.subTest(key=key):
                self.mutate(key,value)
                with self.assertRaises(ValueError): self.compare()
                self.save(self.b,original)

    def test_extra_precision_or_algorithm_knobs_rejected(self):
        original=self.load(self.b)
        for key,value in (('QK_FLASH_COOPMAT','1'),('QK_ATTN_DECODE','ordered'),('QK_FLASH_GEMM','compact'),('QK_FLASH_BATCH','0')):
            env=dict(original['knobs']); env[key]=value; self.mutate('knobs',env)
            with self.assertRaises(ValueError): self.compare()
            self.save(self.b,original)

    def test_truncation_and_unbound_corruption_rejected(self):
        data=self.b.read_bytes(); self.b.write_bytes(data[:-4])
        with self.assertRaises(ValueError): self.compare()
        self.b.write_bytes(bytes([data[0]^1])+data[1:])
        with self.assertRaises(ValueError): self.compare()

    def test_wrong_greedy_is_checked_against_dump(self):
        ids=self.load(self.b)['greedy_ids']; ids[1]=0; self.mutate('greedy_ids',ids)
        self.assertEqual(self.compare()['result'],'FAIL')

    def test_reordered_or_missing_rows_rejected(self):
        rows=self.load(self.b)['row_plan']; rows[1],rows[2]=rows[2],rows[1]; self.mutate('row_plan',rows)
        with self.assertRaises(ValueError): self.compare()

    def test_runtime_and_output_path_can_differ(self):
        self.mutate('seconds',123.)
        self.assertEqual(self.compare()['result'],'PASS')

    def test_policy_order_rejected(self):
        with self.assertRaises(ValueError): mod.compare_paths(self.b,self.a,'vec4')
        with self.assertRaises(ValueError): mod.compare_paths(self.a,self.b,'compact')

if __name__ == '__main__': unittest.main()
