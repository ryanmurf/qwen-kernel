import copy
import json
from pathlib import Path
import unittest

from audit_native_last_head import audit

ROOT = Path(__file__).parent


class AuditTest(unittest.TestCase):
    def fixtures(self):
        return [[json.loads(line) for line in (ROOT / name).read_text().splitlines()]
                for name in ('results-halo-last-head.jsonl', 'results-halo-last-head-reference.jsonl',
                             'results-halo-last-head-profile.jsonl')] + [
                    (ROOT / 'results-halo-last-head-profile.log').read_text()]

    def test_recorded_gate(self):
        result = audit(*self.fixtures())
        self.assertEqual(result['result'], 'PASS')
        self.assertFalse(result['full_model_validation'])
        self.assertFalse(result['default_promotion'])
        self.assertEqual(result['old_abi_exact_id_buffers'], 52)

    def test_rejects_mutated_evidence(self):
        baseline = self.fixtures()
        for mutation in ('missing_case', 'inexact', 'old_id', 'profile_build', 'harness',
                         'shader', 'median', 'sample_order', 'pressure', 'profile_dispatch'):
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(baseline)
                gate, reference, profile, log = data
                if mutation == 'missing_case':
                    gate.pop(next(i for i, row in enumerate(gate) if row['type'] == 'parity'))
                elif mutation == 'inexact':
                    next(r for r in gate if r['type'] == 'parity')['bit_exact'] = False
                elif mutation == 'old_id':
                    next(r for r in reference if r['type'] == 'reference')['all_ids_sha256'][0] = '0'*64
                elif mutation == 'profile_build':
                    profile[0]['library_sha256'] = '0'*64
                elif mutation == 'harness':
                    gate[0]['harness_sha256'] = '0'*64
                elif mutation == 'shader':
                    reference[0]['shader_sha256']['qwen4_gemm_q6k.spv'] = '0'*64
                elif mutation == 'median':
                    next(r for r in gate if r['type'] == 'timing')['all_median'] *= 2
                elif mutation == 'sample_order':
                    next(r for r in gate if r['type'] == 'sample')['last_only'] = True
                elif mutation == 'pressure':
                    gate[-1]['cgroup']['memory.events'] = 'high 1'
                elif mutation == 'profile_dispatch':
                    data[3] = log.replace('qwen4_argmax.spv', 'wrong.spv', 1)
                with self.assertRaises(ValueError):
                    audit(*data)


if __name__ == '__main__':
    unittest.main()
