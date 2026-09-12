"""CPU-only API evidence checks for the explicit vec4 experiment."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
batch=module('batch',ROOT/'bench/compare_native_batch_attention.py')
fixtures=module('fixtures',ROOT/'tests/test_compare_native_attention.py')

def gates():
    long={'result':'PASS','batch_attention_bit_exact':True,'paired_commands_match':True,
          'paired_manifest_identity_match':True,'library_sha256':'b'*64,'ids_sha256':'1'*64,
          'numerical_gate':{'result':'PASS','tail_positions':128,'argmax_agree':128,'argmax_flips':0,
                            'relative_rms_max':0,'clean_first_row_relative_rms':0,'clean_first_row_equal':True}}
    prefill={'result':'PASS','experiment':'native-prefill-exact-v1','candidate':'vec4','bit_exact':True,'failures':[],
             'saved_row_counts':{'clean':1,'prefill':32,'tail':181},'partial_prefill_widths':[459],
             'n':16512,'tail':181,'ctx':32768,'library_sha256':'b'*64,'ids_sha256':'1'*64,
             'dump_sha256':['d'*64]*2,'manifest_sha256':['c'*64,'e'*64],
             'helper_sha256':{'helper':'f'*64},'shaders':{'a':'a'*64},
             'model':{'shards':[{'path':'m','size':100,'mtime_ns':42}]}}
    return json.dumps(long).encode(),json.dumps(prefill).encode()

def fixture(policy, wires=None):
    wires=wires or gates(); rows=fixtures.fixture('serial'); meta=rows[0]['metadata']
    env={**batch.FIXED_ENV,'QK_SHADER_DIR':'/shaders','QK_GGUF':'m','QK_FLASH_ATTN_BATCH':policy}
    meta['actual_qk_environment']=env
    meta['server_command']=['launch',f'--unit={policy}']+[f'{k}={env[k]}' for k in
          ('QK_FLASH_ATTN_BATCH','QK_FLASH_GEMM','QK_ATTN_DECODE','QK_FLASH_COOPMAT',
           'QK_PLE_PREFETCH','QK_PLE_ROW_PREFETCH','QK_PREFILL_CHUNK','QK_ATTN_CHUNK')]
    meta.update(experiment='halo-vec4-api-v1',actual_batch_attention_policy=policy,
                actual_batch_attention_announcement=('baseline' if policy=='baseline' else 'vec4-F32-QB8 (coopmat takes precedence)'),
                vec4_dispatch_evidence=([] if policy=='baseline' else
                    ['native vec4 batch attention first dispatch: fa_attn_batch_vec4.spv base=0 rows=309 qbase=0 tile=309 QB=8']),
                actual_gemm_announcement='baseline',compact_dispatch_evidence=[],
                prefill_math='F32 scalar',context=32768,slots=1,prefill_chunk=512,
                http_passed=sorted(batch.HTTP_TESTS),http_stream_coherent=True,
                source_sha256={'runner':'9'*64},server_sha256='8'*64,
                long_gate_sha256=hashlib.sha256(wires[0]).hexdigest(),prefill_gate_sha256=hashlib.sha256(wires[1]).hexdigest())
    fixtures.rehash(rows); return rows

class BatchComparisonTests(unittest.TestCase):
    def test_valid_without_mutation(self):
        pair=[fixture('baseline'),fixture('vec4')]; saved=copy.deepcopy(pair)
        result=batch.compare(*pair,*gates())
        self.assertTrue(result['comparison_valid'] and result['outputs_match'])
        self.assertFalse(result['default_promotion']); self.assertEqual(pair,saved)

    def test_binding_both_gates(self):
        a,b=gates()
        for aa,bb in ((a+b' ',b),(a,b+b' ')):
            with self.assertRaises(ValueError): batch.compare(fixture('baseline'),fixture('vec4'),aa,bb)

    def test_partial_and_prefill_gate_cannot_be_omitted(self):
        a,b=gates(); original=json.loads(b)
        for key,value in [('saved_row_counts',{'clean':1,'prefill':0,'tail':181}),('partial_prefill_widths',[]),
                          ('bit_exact',False),('result','FAIL'),('candidate','compact'),('tail',128),
                          ('failures',[{}]),('dump_sha256',['a'*64,'b'*64]),('manifest_sha256',[]),
                          ('library_sha256','f'*64),('ids_sha256','e'*64),('helper_sha256',{})]:
            g=copy.deepcopy(original); g[key]=value; wires=(a,json.dumps(g).encode())
            with self.subTest(key=key),self.assertRaises(ValueError):
                batch.compare(fixture('baseline',wires),fixture('vec4',wires),*wires)

    def test_previous_long_gate_still_required(self):
        a,b=gates(); original=json.loads(a)
        for key,value in [('result','FAIL'),('batch_attention_bit_exact',False),('paired_commands_match',False)]:
            g=copy.deepcopy(original); g[key]=value; wires=(json.dumps(g).encode(),b)
            with self.assertRaises(ValueError): batch.compare(fixture('baseline',wires),fixture('vec4',wires),*wires)
        g=copy.deepcopy(original); g['numerical_gate']['relative_rms_max']=1e-9; wires=(json.dumps(g).encode(),b)
        with self.assertRaises(ValueError): batch.compare(fixture('baseline',wires),fixture('vec4',wires),*wires)

    def test_environment_policy_and_dispatch(self):
        for key,value in [('QK_FLASH_COOPMAT','1'),('QK_FLASH_GEMM','compact'),('QK_ATTN_DECODE','ordered'),
                          ('QK_DEVICE_NAME','NAVI31'),('QK_FLASH_ATTN_BATCH','baseline'),('QK_FLASH_TIMING','1')]:
            b=fixture('vec4'); b[0]['metadata']['actual_qk_environment'][key]=value; fixtures.rehash(b)
            with self.subTest(key=key),self.assertRaises(ValueError): batch.compare(fixture('baseline'),b,*gates())
        for key,value in [('vec4_dispatch_evidence',[]),('actual_batch_attention_announcement','baseline'),
                          ('actual_gemm_announcement','compact'),('compact_dispatch_evidence',['x']),
                          ('prefill_math','F16'),('kv_precision','F16'),('http_passed',[]),('mtp',True),
                          ('source_sha256',{}),('device_pci','0000:68:00.0')]:
            b=fixture('vec4'); b[0]['metadata'][key]=value; fixtures.rehash(b)
            with self.subTest(key=key),self.assertRaises(ValueError): batch.compare(fixture('baseline'),b,*gates())

    def test_duplicate_overrides_rejected_even_if_both_match(self):
        for key in ('QK_FLASH_ATTN_BATCH','QK_ATTN_DECODE','QK_FLASH_GEMM','QK_FLASH_COOPMAT'):
            pair=[fixture('baseline'),fixture('vec4')]
            for rows in pair: rows[0]['metadata']['server_command'].append(key+'=wrong'); fixtures.rehash(rows)
            with self.assertRaises(ValueError): batch.compare(*pair,*gates())

    def test_mismatched_sources_shaders_outputs_and_cells(self):
        for key,value in [('source_sha256',{'runner':'7'*64}),('shaders',{'a':'8'*64}),('server_sha256','7'*64),('slots',2)]:
            b=fixture('vec4'); b[0]['metadata'][key]=value; fixtures.rehash(b)
            with self.assertRaises(ValueError): batch.compare(fixture('baseline'),b,*gates())
        for index,key,value in [(1,'output_tokens',[17]),(2,'output_text_sha256','0'*64),(2,'server_cached_tokens',1),
                                (2,'streamed_tokens',127),(2,'ttft_seconds',float('nan'))]:
            b=fixture('vec4'); b[index][key]=value
            with self.assertRaises(ValueError): batch.compare(fixture('baseline'),b,*gates())
        with self.assertRaises(ValueError): batch.compare(fixture('baseline'),fixture('vec4')[:-1],*gates())

    def test_prior_comparator_not_relaxed(self):
        with self.assertRaises(ValueError): batch.ab.compare(fixture('baseline'),fixture('vec4'))

if __name__=='__main__': unittest.main()
