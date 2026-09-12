#!/usr/bin/env python3
"""Audit matched baseline/vec4 API runs bound to BOTH full-model gates.

Read-only; does not load a model. Scalar F32, serial decode and baseline
GEMM in both. A valid comparison is not an automatic default promotion.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

SPEC = importlib.util.spec_from_file_location('attention_ab', Path(__file__).with_name('compare_native_attention.py'))
ab = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ab)
require = ab.require
HTTP_TESTS = {'teacher_forced_http', 'claude_text', 'claude_xml_tool', 'claude_tool_roundtrip',
              'claude_stream', 'prefill_cancel_then_request', 'concurrent_isolation', 'invalid_token'}
FIXED_ENV = {'QK_ATTN_DECODE': 'serial', 'QK_DEVICE_NAME': 'STRIX_HALO', 'QK_FLASH_GEMM': 'baseline',
             'QK_NATIVE_FLASH': '1', 'QK_FLASH_COOPMAT': '0', 'QK_PLE_PREFETCH': '0',
             'QK_PREFILL_CHUNK': '512', 'QK_ATTN_CHUNK': '256', 'QK_PLE_ROW_PREFETCH': '1',
             'QK_REASONING_EFFORT': 'xhigh'}

def valid_hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def normalized_metadata(metadata, policy):
    require(policy in ('baseline','vec4') and metadata['experiment'] == 'halo-vec4-api-v1', 'wrong experiment/policy')
    env = metadata['actual_qk_environment']
    expected = {**FIXED_ENV, 'QK_FLASH_ATTN_BATCH': policy,
                'QK_GGUF': metadata['model']['shards'][0]['path']}
    require(set(env) == set(expected) | {'QK_SHADER_DIR'} and
            all(env.get(k) == v for k,v in expected.items()) and
            isinstance(env['QK_SHADER_DIR'],str) and Path(env['QK_SHADER_DIR']).is_absolute(), 'ungated environment')
    require(metadata['actual_batch_attention_policy'] == policy and
            metadata['actual_batch_attention_announcement'] == ('baseline' if policy == 'baseline' else
                'vec4-F32-QB8 (coopmat takes precedence)'), 'batch policy announcement mismatch')
    require(metadata['actual_gemm_announcement'] == 'baseline' and metadata['compact_dispatch_evidence'] == [], 'GEMM not baseline')
    require(metadata['prefill_math'] == 'F32 scalar' and metadata['context'] == 32768 and
            metadata['slots'] == 1 and metadata['prefill_chunk'] == 512, 'ungated configuration')
    require(sorted(metadata['http_passed']) == sorted(HTTP_TESTS) and metadata['http_stream_coherent'] is True, 'incomplete HTTP suite')
    require(metadata['source_sha256'] and all(valid_hash(v) for v in metadata['source_sha256'].values()) and
            valid_hash(metadata['server_sha256']), 'missing source/server identities')
    dispatch = metadata['vec4_dispatch_evidence']
    require(isinstance(dispatch,list), 'invalid dispatch evidence')
    if policy == 'baseline': require(dispatch == [], 'baseline ran vec4')
    else:
        require(len(dispatch) == 1 and isinstance(dispatch[0],str), 'missing/duplicate vec4 dispatch')
        match = re.fullmatch(r'native vec4 batch attention first dispatch: fa_attn_batch_vec4\.spv base=0 rows=(\d+) qbase=0 tile=(\d+) QB=8',dispatch[0])
        require(match is not None and 2 <= int(match[1]) <= 512 and 1 <= int(match[2]) <= int(match[1]), 'unexpected vec4 dispatch')
    for key in ('QK_FLASH_ATTN_BATCH','QK_FLASH_GEMM','QK_ATTN_DECODE','QK_FLASH_COOPMAT',
                'QK_PLE_PREFETCH','QK_PLE_ROW_PREFETCH','QK_PREFILL_CHUNK','QK_ATTN_CHUNK'):
        require([s for s in metadata['server_command'] if s.startswith(key+'=')] == [f'{key}={expected[key]}'], 'launch override mismatch')
    value = ab.normalized_metadata(metadata,'serial')
    value['server_command'] = ['QK_FLASH_ATTN_BATCH=POLICY' if s.startswith('QK_FLASH_ATTN_BATCH=') else s for s in value['server_command']]
    value['actual_qk_environment']['QK_FLASH_ATTN_BATCH'] = 'POLICY'
    value['actual_batch_attention_policy'] = 'POLICY'; value['actual_batch_attention_announcement'] = 'POLICY'
    value['vec4_dispatch_evidence'] = 'VERIFIED'
    return value

def validate_gates(long_bytes, prefill_bytes):
    long = json.loads(long_bytes); prefill = json.loads(prefill_bytes)
    require(long['result'] == 'PASS' and long['batch_attention_bit_exact'] is True and
            long['paired_commands_match'] is True and long['paired_manifest_identity_match'] is True, 'long gate failed')
    numeric = long['numerical_gate']
    require(numeric['result'] == 'PASS' and numeric['tail_positions'] == 128 and numeric['argmax_agree'] == 128 and
            numeric['argmax_flips'] == 0 and numeric['relative_rms_max'] == 0 and
            numeric['clean_first_row_relative_rms'] == 0 and numeric['clean_first_row_equal'] is True, 'non-exact long gate')
    require(prefill['result'] == 'PASS' and prefill['experiment'] == 'native-prefill-exact-v1' and
            prefill['candidate'] == 'vec4' and prefill['bit_exact'] is True and prefill['failures'] == [], 'prefill gate failed')
    require(prefill['saved_row_counts'] == {'clean':1,'prefill':32,'tail':181} and
            prefill['partial_prefill_widths'] == [459] and prefill['n'] == 16512 and
            prefill['tail'] == 181 and prefill['ctx'] == 32768, 'missing partial-prefix coverage')
    require(len(prefill['dump_sha256']) == 2 and len(set(prefill['dump_sha256'])) == 1 and
            all(valid_hash(v) for v in prefill['dump_sha256']) and len(prefill['manifest_sha256']) == 2 and
            all(valid_hash(v) for v in prefill['manifest_sha256']), 'invalid dump/manifest binding')
    require(prefill['library_sha256'] == long['library_sha256'] and prefill['ids_sha256'] == long['ids_sha256'], 'full-model gate identity mismatch')
    require(prefill['helper_sha256'] and all(valid_hash(v) for v in prefill['helper_sha256'].values()), 'missing helper identity')
    long_sha, prefill_sha = hashlib.sha256(long_bytes).hexdigest(), hashlib.sha256(prefill_bytes).hexdigest()
    return prefill, long_sha, prefill_sha

def compare(baseline, vec4, long_bytes, prefill_bytes):
    prefill, long_sha, prefill_sha = validate_gates(long_bytes, prefill_bytes)
    for rows in (baseline,vec4):
        require(rows, 'empty API run'); meta = rows[0]['metadata']
        require(meta['long_gate_sha256'] == long_sha and meta['prefill_gate_sha256'] == prefill_sha and
                meta['library_sha256'] == prefill['library_sha256'] and meta['shaders'] == prefill['shaders'] and
                meta['model'] == prefill['model'], 'API build differs from bound gates')
    result = ab.compare_modes(baseline,vec4,modes=('baseline','vec4'),normalize=normalized_metadata)
    result.update(experiment='halo-vec4-api-v1',long_gate_sha256=long_sha,prefill_gate_sha256=prefill_sha,
                  default_promotion=False)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path); parser.add_argument('vec4',type=Path)
    parser.add_argument('--long-gate',required=True,type=Path); parser.add_argument('--prefill-gate',required=True,type=Path)
    args = parser.parse_args()
    try:
        load = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        print(json.dumps(compare(load(args.baseline),load(args.vec4),args.long_gate.read_bytes(),args.prefill_gate.read_bytes()),indent=2,allow_nan=False))
    except (OSError,ValueError,KeyError,TypeError) as error: parser.exit(1,f'error: {error}\n')

if __name__ == '__main__': main()
