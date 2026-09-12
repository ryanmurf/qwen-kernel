import copy
import unittest

from audit_native_last_head_resources import GIB, memory_bytes, validate_resources


def fixture():
    unit = 'qk-last-head-http-all-test'
    props = dict(MainPID='42', SubState='running', ExecMainStatus='0',
                 MemoryPeak='123456', MemorySwapPeak='0')
    admission = dict(admitted=True, minimum_available_gib=24, maximum_other_memory_gib=12,
                     available_kib=28 * 2**20, anonymous_shared_and_swap_kib=2 * 2**20)
    controller = dict(mode='all', unit=unit, error=None, identities_unchanged=True,
                      default_promotion=False, final_unit=dict(props), admission=admission,
                      actual_environment={'QK_FLASH_PREFILL_LAST': '0'},
                      start_utc='2026-09-12T00:00:00Z', end_utc='2026-09-12T00:00:07Z', cleanup={})
    for key in ('http', 'benchmark', 'audit', 'stop', 'observer'):
        controller[key + '_returncode'] = 0
    row = dict(elapsed=.1, unit=unit, properties=props, memory={'MemAvailable': 26 * GIB},
               temperatures={'temp1_input': '72000'}, violation=None,
               events={'max': '0', 'oom': '0', 'oom_kill': '0'},
               devices={'1': {'drm-pdev': '0000:c1:00.0', 'drm-engine-gfx': '100 ns'},
                        '2': {'drm-pdev': '0000:68:00.0', 'drm-memory-vram': '12 KiB',
                              'drm-memory-gtt': '2 MiB'}})
    rows = [row, copy.deepcopy(row), copy.deepcopy(row)]
    rows[1]['elapsed'] = 3.1
    rows[2].update(elapsed=6.1, devices={}, events={})
    rows[2]['properties'].update(SubState='dead', MainPID='0',
                                  MemoryPeak=str(2**64 - 1), MemorySwapPeak=str(2**64 - 1))
    return controller, rows


class ResourceAuditTest(unittest.TestCase):
    def test_complete_run(self):
        result = validate_resources(*fixture())
        self.assertEqual(result['result'], 'PASS')
        self.assertEqual(result['cgroup_swap_peak_bytes'], 0)
        self.assertEqual(result['cgroup_memory_peak_bytes'], 123456)
        self.assertEqual(result['external_gpu_observations']['0000:68:00.0']['gtt_peak_bytes'], 2**21)

    def test_memory_units(self):
        for text, expected in [('0', 0), ('4 B', 4), ('2 KiB', 2048), ('3 MiB', 3 * 2**20)]:
            self.assertEqual(memory_bytes(text), expected)
        for text in ('-1 KiB', '2 GB', 'NaN KiB', '1 KiB extra', ''):
            with self.assertRaises(ValueError):
                memory_bytes(text)

    def test_rejects_incomplete_or_unsafe_run(self):
        for case in ('error', 'identity', 'returncode', 'policy', 'admission', 'stopped_early',
                     'no_stop', 'gap', 'reorder', 'alarm', 'low_ram', 'temperature', 'pid',
                     'missing_events', 'oom', 'xtx_compute', 'xtx_memory', 'missing_gpu_memory',
                     'no_compute', 'swap', 'cgroup_memory'):
            with self.subTest(case=case):
                controller, rows = fixture()
                if case == 'error': controller['error'] = 'failure'
                elif case == 'identity': controller['identities_unchanged'] = False
                elif case == 'returncode': controller['http_returncode'] = 1
                elif case == 'policy': controller['actual_environment']['QK_FLASH_PREFILL_LAST'] = '1'
                elif case == 'admission': controller['admission']['minimum_available_gib'] = 8
                elif case == 'stopped_early': controller['final_unit']['SubState'] = 'dead'
                elif case == 'no_stop': rows.pop()
                elif case == 'gap': rows[2]['elapsed'] = 30
                elif case == 'reorder': rows[1]['elapsed'] = 0
                elif case == 'alarm': rows[0]['violation'] = 'alarm'
                elif case == 'low_ram': rows[0]['memory']['MemAvailable'] = 7 * GIB
                elif case == 'temperature': rows[0]['temperatures']['temp1_input'] = '93000'
                elif case == 'pid': rows[0]['properties']['MainPID'] = '43'
                elif case == 'missing_events': rows[0]['events'] = {}
                elif case == 'oom': rows[0]['events']['oom_kill'] = '1'
                elif case == 'xtx_compute': rows[0]['devices']['2']['drm-engine-gfx'] = '1 ns'
                elif case == 'xtx_memory': rows[0]['devices']['2']['drm-memory-gtt'] = '1 GiB'
                elif case == 'missing_gpu_memory': del rows[0]['devices']['2']['drm-memory-gtt']
                elif case == 'no_compute':
                    for row in rows: row['devices'] = {}
                elif case == 'swap': rows[0]['properties']['MemorySwapPeak'] = str(GIB)
                elif case == 'cgroup_memory': rows[0]['properties']['MemoryPeak'] = str(33 * GIB)
                with self.assertRaises(ValueError):
                    validate_resources(controller, rows)


if __name__ == '__main__':
    unittest.main()
