"""CPU-only audit tests; synthetic fixtures are not benchmark evidence."""
import copy
import unittest

from audit_native_ktranspose import CASES, MODES, ORDER, validate_cells, validate_resources


def cells():
    return [dict(n0=case[0], n1=case[1], q_scale=case[2], mode=MODES[mode], order=order,
                 gpu_us=100/(mode+1), fp64_max_abs=1e-6, nonfinite=0, padding_changes=0,
                 baseline_bit_mismatches=0, fp64_misses=0, result='PASS')
            for case in CASES for order, mode in enumerate(ORDER)]


def telemetry():
    props = dict(MainPID='123', SubState='running', ExecMainStatus='0',
                 MemoryPeak=str(512*2**20), MemorySwapPeak='0')
    events = {key:'0' for key in ('low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill')}
    row = dict(elapsed=0.01, unit='test-unit', properties=props,
               memory={'MemAvailable':20*2**30}, temperatures={'temp1_input':'70000'},
               devices={'1':{'drm-pdev':'0000:c1:00.0', 'drm-engine-gfx':'100 ns'},
                        '2':{'drm-pdev':'0000:68:00.0', 'drm-memory-vram':'12 KiB',
                             'drm-memory-gtt':'2 MiB'}}, events=events, violation=None)
    rows = [copy.deepcopy(row) for _ in range(3)]
    rows[1]['elapsed'] = 3.02
    rows[2]['elapsed'] = 6.03
    rows[2]['properties'].update(SubState='exited', MainPID='0')
    command = ['systemd-run', '--user', '--service-type=exec']
    for flag in ('MemoryHigh=1G', 'MemoryMax=2G', 'MemorySwapMax=0', 'NoNewPrivileges=yes',
                 'LimitCORE=0', 'RuntimeMaxSec=600', 'RemainAfterExit=yes'):
        command += ['-p', flag]
    command += ['/path/operator', '4']
    controller = dict(result='PASS', error=None, identities_unchanged=True, stop_returncode=0,
        observer_returncode=0, full_model=False, default_promotion=False,
        available_before_bytes=20*2**30, command=command, unit='test-unit',
        final_unit=dict(SubState='exited', ExecMainStatus='0', MemoryPeak=str(512*2**20),
                        MemorySwapPeak='0'))
    return controller, rows


class KTransposeAuditTests(unittest.TestCase):
    def test_complete_order_and_ratios(self):
        summary = validate_cells(cells())
        self.assertEqual(len(summary), 64)
        for row in summary:
            self.assertEqual(row['gpu_us']['cells'], 2)
            self.assertAlmostEqual(row['speedup'], MODES.index(row['mode'])+1)

    def test_bad_cell_rejected(self):
        for field, value in [('gpu_us', 0), ('gpu_us', float('nan')), ('gpu_us', float('inf')),
                ('fp64_max_abs', float('nan')), ('result', 'FAIL'), ('nonfinite', 1),
                ('padding_changes', 1), ('baseline_bit_mismatches', 1), ('fp64_misses', 1),
                ('n0', 99), ('order', 3), ('mode', 'unknown')]:
            with self.subTest(field=field, value=value):
                rows = cells()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    validate_cells(rows)
        with self.assertRaises(ValueError):
            validate_cells(cells()[:-1])
        with self.assertRaises(ValueError):
            validate_cells(list(reversed(cells())))

    def test_resource_receipt(self):
        result = validate_resources(*telemetry())
        self.assertEqual(result['samples'], 3)
        self.assertEqual(result['swap_peak_bytes'], 0)
        self.assertEqual(result['maximum_temperature_c'], 70)

    def test_resource_corruption(self):
        changes = [
            lambda c, r: c.update(default_promotion=True),
            lambda c, r: c.update(full_model=True),
            lambda c, r: c.update(available_before_bytes=11*2**30),
            lambda c, r: c.update(identities_unchanged=False),
            lambda c, r: c['command'].__setitem__(c['command'].index('MemorySwapMax=0'), 'MemorySwapMax=1G'),
            lambda c, r: c['final_unit'].update(MemorySwapPeak='4096'),
            lambda c, r: r[1].update(elapsed=40),
            lambda c, r: r[1].update(violation='alarm'),
            lambda c, r: r[1]['memory'].update(MemAvailable=7*2**30),
            lambda c, r: r[1]['temperatures'].update(temp1_input='93000'),
            lambda c, r: r[1]['properties'].update(MainPID='456'),
            lambda c, r: r[1]['properties'].update(MemorySwapPeak='4096'),
            lambda c, r: r[1]['properties'].update(MemoryPeak=str(3*2**30)),
            lambda c, r: r[1]['events'].update(oom_kill='1'),
            lambda c, r: r[1]['events'].update(oom_group_kill='1'),
            lambda c, r: r[1]['devices']['2'].update({'drm-engine-gfx':'1 ns'}),
            lambda c, r: r[1]['devices']['2'].update({'drm-memory-gtt':'9 MiB'}),
            lambda c, r: r[2]['properties'].update(SubState='failed'),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                controller, rows = telemetry()
                change(controller, rows)
                with self.assertRaises(ValueError):
                    validate_resources(controller, rows)


if __name__ == '__main__':
    unittest.main()
