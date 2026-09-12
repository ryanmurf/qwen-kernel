"""CPU-only GEMM evidence checks; no GPU/model/network."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gemm_ab', ROOT / 'bench/compare_native_gemm.py')
gemm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gemm)
SPEC = importlib.util.spec_from_file_location('attention_fixture', ROOT / 'tests/test_compare_native_attention.py')
attention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attention)


def gate_bytes():
    return json.dumps({'result': 'PASS', 'gemm_bit_exact': True, 'paired_commands_match': True,
                       'paired_manifest_identity_match': True, 'library_sha256': 'b' * 64,
                       'numerical_gate': {'result': 'PASS', 'tail_positions': 128, 'argmax_agree': 128,
                                          'argmax_flips': 0, 'relative_rms_max': 0,
                                          'clean_first_row_relative_rms': 0, 'clean_first_row_equal': True},
                       'evidence': [{'policy': p, 'dump_sha256': 'd' * 64, 'abi_greedy_agree': 128}
                                    for p in ('baseline', 'compact')]}).encode()


def fixture(policy, gate=None):
    gate = gate if gate is not None else gate_bytes()
    rows = attention.fixture('serial')
    meta = rows[0]['metadata']
    env = meta['actual_qk_environment']
    env.update(QK_FLASH_GEMM=policy, QK_NATIVE_FLASH='1', QK_FLASH_COOPMAT='0',
               QK_PLE_PREFETCH='0', QK_PREFILL_CHUNK='512')
    meta['server_command'] += [f'{k}={env[k]}' for k in
                               ('QK_FLASH_GEMM', 'QK_FLASH_COOPMAT', 'QK_PLE_PREFETCH', 'QK_PREFILL_CHUNK')]
    meta.update(experiment='halo-compact-gemm-v1', actual_gemm_policy=policy,
                actual_gemm_announcement=('baseline' if policy == 'baseline' else
                                         'compact-F32 (shape-selected; coopmat takes precedence)'),
                compact_dispatch_evidence=([] if policy == 'baseline' else
                    ['native compact GEMM first dispatch: qwen4_gemm_compact_q5_1.spv M=10240 K=320 rows=512']),
                prefill_math='F32 scalar', context=32768, slots=1, prefill_chunk=512,
                http_passed=sorted(gemm.HTTP_TESTS), http_stream_coherent=True,
                server_sha256='a' * 64, source_sha256={'runner': 'e' * 64},
                long_gate_sha256=hashlib.sha256(gate).hexdigest())
    rows[1]['prefill_plus_one_token_seconds'] = 10 if policy == 'baseline' else 9
    attention.rehash(rows)
    return rows


class GemmComparisonTests(unittest.TestCase):
    def test_matching_without_mutation(self):
        pair = [fixture('baseline'), fixture('compact')]
        saved = copy.deepcopy(pair)
        result = gemm.compare(*pair, gate_bytes())
        self.assertTrue(result['gemm_bit_exact'])
        self.assertTrue(result['exploratory'])
        self.assertFalse(result['default_promotion'])
        self.assertEqual(result['rows'][0]['prefill_plus_one_token_seconds']['compact_over_baseline'], .9)
        self.assertEqual(pair, saved)

    def test_gate_identity(self):
        pair = [fixture('baseline'), fixture('compact')]
        with self.assertRaisesRegex(ValueError, 'gate/build identity'):
            gemm.compare(*pair, gate_bytes() + b' ')
        pair[1][0]['metadata']['library_sha256'] = 'e' * 64
        attention.rehash(pair[1])
        with self.assertRaisesRegex(ValueError, 'gate/build identity'):
            gemm.compare(*pair, gate_bytes())

    def test_failed_or_incomplete_gate(self):
        mutations = [lambda g: g.update(result='FAIL'), lambda g: g.update(gemm_bit_exact=False),
                     lambda g: g.update(paired_commands_match=False),
                     lambda g: g.update(paired_manifest_identity_match=False),
                     lambda g: g['numerical_gate'].update(relative_rms_max=1e-8),
                     lambda g: g['numerical_gate'].update(argmax_flips=1),
                     lambda g: g['numerical_gate'].update(tail_positions=127),
                     lambda g: g['numerical_gate'].update(clean_first_row_equal=False),
                     lambda g: g['evidence'][1].update(dump_sha256='e' * 64),
                     lambda g: g['evidence'][1].update(abi_greedy_agree=127),
                     lambda g: g.update(evidence=[])]
        for mutate in mutations:
            gate = json.loads(gate_bytes())
            mutate(gate)
            wire = json.dumps(gate).encode()
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                gemm.compare(fixture('baseline', wire), fixture('compact', wire), wire)

    def test_mode_environment_and_dispatch(self):
        mutations = [lambda m: m.update(actual_gemm_policy='baseline'),
                     lambda m: m.update(actual_gemm_announcement='baseline'),
                     lambda m: m.update(actual_decode_attention='ordered-F32'),
                     lambda m: m['actual_qk_environment'].update(QK_ATTN_DECODE='ordered'),
                     lambda m: m['actual_qk_environment'].update(QK_FLASH_COOPMAT='1'),
                     lambda m: m['actual_qk_environment'].update(QK_FLASH_GEMM='baseline'),
                     lambda m: m.update(compact_dispatch_evidence=[]),
                     lambda m: m['compact_dispatch_evidence'].append(m['compact_dispatch_evidence'][0]),
                     lambda m: m.update(compact_dispatch_evidence=['native compact GEMM first dispatch: wrong']),
                     lambda m: m.update(prefill_math='F16'), lambda m: m.update(kv_precision='F16'),
                     lambda m: m.update(mtp=True), lambda m: m.update(device_pci='0000:68:00.0')]
        for mutate in mutations:
            rows = fixture('compact')
            mutate(rows[0]['metadata'])
            attention.rehash(rows)
            with self.subTest(metadata=rows[0]['metadata']), self.assertRaises(ValueError):
                gemm.compare(fixture('baseline'), rows, gate_bytes())

    def test_duplicate_launch_overrides_even_in_both(self):
        for key, value in [('QK_FLASH_GEMM', 'compact'), ('QK_ATTN_DECODE', 'ordered'),
                           ('QK_FLASH_COOPMAT', '1'), ('QK_PLE_PREFETCH', '1'), ('QK_PREFILL_CHUNK', '256')]:
            pair = [fixture('baseline'), fixture('compact')]
            for rows in pair:
                rows[0]['metadata']['server_command'].append(f'{key}={value}')
                attention.rehash(rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                gemm.compare(*pair, gate_bytes())

    def test_full_config_and_http_evidence(self):
        for key, value in [('http_passed', []), ('http_stream_coherent', False), ('source_sha256', {}),
                           ('source_sha256', {'runner': 'e' * 63}), ('source_sha256', {'runner': 'f' * 64}),
                           ('shaders', {'a': 'f' * 64}), ('server_sha256', 'f' * 64), ('slots', 2),
                           ('experiment', 'attention'), ('extra_unmatched_knob', True)]:
            rows = fixture('compact')
            rows[0]['metadata'][key] = value
            attention.rehash(rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                gemm.compare(fixture('baseline'), rows, gate_bytes())

    def test_rows_and_hash_checks_reused(self):
        changes = [(1, 'output_tokens', [17]), (2, 'output_token_sha256', 'a' * 64),
                   (2, 'server_cached_tokens', 8192), (2, 'coherent_counting_prefix', False),
                   (2, 'streamed_tokens', 127), (2, 'ttft_seconds', float('nan'))]
        for index, key, value in changes:
            rows = fixture('compact')
            rows[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                gemm.compare(fixture('baseline'), rows, gate_bytes())
        with self.assertRaises(ValueError):
            gemm.compare(fixture('baseline'), fixture('compact')[:-1], gate_bytes())

    def test_attention_comparator_not_relaxed(self):
        with self.assertRaises(ValueError):
            gemm.ab.compare(fixture('baseline'), fixture('compact'))


if __name__ == '__main__':
    unittest.main()
