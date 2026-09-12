#!/usr/bin/env python3
"""Read-only matched baseline/compact GEMM API audit, with a bound logits gate.

Serial attention, scalar F32 prefill, F32 KV and no MTP are required in both.
This is an explicit GEMM experiment, not a relaxation of the attention audit.
Recorded HTTP checks and counting outputs are not general quality evaluation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

SPEC = importlib.util.spec_from_file_location('attention_ab', Path(__file__).with_name('compare_native_attention.py'))
ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ab)
require = ab.require
HTTP_TESTS = {'teacher_forced_http', 'claude_text', 'claude_xml_tool', 'claude_tool_roundtrip',
              'claude_stream', 'prefill_cancel_then_request', 'concurrent_isolation', 'invalid_token'}


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def normalized_metadata(metadata, policy):
    require(policy in ('baseline', 'compact'), 'unsupported GEMM policy')
    require(metadata['experiment'] == 'halo-compact-gemm-v1', 'wrong experiment')
    env = metadata['actual_qk_environment']
    expected = {'QK_FLASH_GEMM': policy, 'QK_ATTN_DECODE': 'serial', 'QK_FLASH_COOPMAT': '0',
                'QK_NATIVE_FLASH': '1', 'QK_PLE_PREFETCH': '0', 'QK_PREFILL_CHUNK': '512'}
    require(all(env.get(k) == v for k, v in expected.items()), 'GEMM/attention/precision environment mismatch')
    require(metadata['actual_gemm_policy'] == policy, 'recorded GEMM policy mismatch')
    require(metadata['actual_gemm_announcement'] == ('baseline' if policy == 'baseline' else
            'compact-F32 (shape-selected; coopmat takes precedence)'), 'GEMM announcement mismatch')
    require(metadata['prefill_math'] == 'F32 scalar' and metadata['context'] == 32768
            and metadata['slots'] == 1 and metadata['prefill_chunk'] == 512, 'wrong gated configuration')
    require(sorted(metadata['http_passed']) == sorted(HTTP_TESTS)
            and metadata['http_stream_coherent'] is True, 'incomplete HTTP checks')
    require(metadata['source_sha256'] and all(valid_hash(v) for v in metadata['source_sha256'].values())
            and valid_hash(metadata['server_sha256']), 'missing controller/test/server identity')
    dispatch = metadata['compact_dispatch_evidence']
    require(isinstance(dispatch, list), 'invalid dispatch evidence')
    if policy == 'baseline':
        require(dispatch == [], 'baseline unexpectedly dispatched compact GEMM')
    else:
        # PLE is the first eligible projection in this model and HTTP suite.
        require(len(dispatch) == 1 and isinstance(dispatch[0], str), 'missing/duplicate compact dispatch')
        match = re.fullmatch(r'native compact GEMM first dispatch: qwen4_gemm_compact_q5_1\.spv M=10240 K=320 rows=(\d+)', dispatch[0])
        require(match is not None and 64 <= int(match[1]) <= 512, 'unexpected compact dispatch shape')
    command = metadata['server_command']
    for key in ('QK_FLASH_GEMM', 'QK_ATTN_DECODE', 'QK_FLASH_COOPMAT', 'QK_PLE_PREFETCH', 'QK_PREFILL_CHUNK'):
        require([a for a in command if a.startswith(key + '=')] == [f'{key}={expected[key]}'],
                f'{key} launch mismatch or duplicate override')
    # Reuse the unchanged Halo/serial/F32/no-MTP validation. Only the GEMM
    # selector, its announcement and verified dispatch evidence may differ.
    value = ab.normalized_metadata(metadata, 'serial')
    value['server_command'] = ['QK_FLASH_GEMM=POLICY' if a.startswith('QK_FLASH_GEMM=') else a
                               for a in value['server_command']]
    value['actual_qk_environment']['QK_FLASH_GEMM'] = 'POLICY'
    value['actual_gemm_policy'] = 'POLICY'
    value['actual_gemm_announcement'] = 'POLICY'
    value['compact_dispatch_evidence'] = 'VERIFIED'
    return value


def compare(baseline, compact, gate_bytes):
    gate = json.loads(gate_bytes)
    require(gate['result'] == 'PASS' and gate['gemm_bit_exact'] is True
            and gate['paired_commands_match'] is True and gate['paired_manifest_identity_match'] is True,
            'same-build bit-exact long gate did not pass')
    numeric = gate['numerical_gate']
    require(numeric['result'] == 'PASS' and numeric['tail_positions'] == 128
            and numeric['argmax_agree'] == 128 and numeric['argmax_flips'] == 0
            and numeric['relative_rms_max'] == 0 and numeric['clean_first_row_relative_rms'] == 0
            and numeric['clean_first_row_equal'] is True, 'incomplete/non-exact logit gate')
    evidence = gate['evidence']
    require(len(evidence) == 2 and {e['policy'] for e in evidence} == {'baseline', 'compact'}
            and len({e['dump_sha256'] for e in evidence}) == 1
            and all(valid_hash(e['dump_sha256']) and e['abi_greedy_agree'] == 128 for e in evidence),
            'inconsistent gate evidence')
    gate_sha = hashlib.sha256(gate_bytes).hexdigest()
    for rows in (baseline, compact):
        require(rows, 'empty run')
        metadata = rows[0]['metadata']
        require(metadata['long_gate_sha256'] == gate_sha
                and metadata['library_sha256'] == gate['library_sha256'], 'gate/build identity mismatch')
    result = ab.compare_modes(baseline, compact, modes=('baseline', 'compact'), normalize=normalized_metadata)
    result.update(experiment='halo-compact-gemm-v1', long_gate_sha256=gate_sha,
                  gemm_bit_exact=True, default_promotion=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('compact', type=Path)
    parser.add_argument('--gate', required=True, type=Path)
    args = parser.parse_args()
    try:
        load = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        print(json.dumps(compare(load(args.baseline), load(args.compact), args.gate.read_bytes()),
                         indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
