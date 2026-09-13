#!/usr/bin/env python3
"""Read-only audit of the isolated, persistent-K-layout operator experiment.

GPU intervals exclude uploads and construction/maintenance of the extra cache.
Passing this audit never promotes a serving configuration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_native_last_head_resources import memory_bytes


MODES = ('baseline', 'transpose', 'pad16', 'pad32')
ORDER = (0, 1, 2, 3, 3, 2, 1, 0)
CASES = [(n, 0, 1) for n in (1, 2, 63, 64, 65, 255, 256, 257, 1025, 16384, 16433, 32768)]
CASES += [(257, 16433, 1), (1025, 0, 16), (257, 0, 128), (127, 0, 1)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_cells(rows):
    expected = [(*case, MODES[mode], order) for case in CASES for order, mode in enumerate(ORDER)]
    actual = [(r['n0'], r['n1'], r['q_scale'], r['mode'], r['order']) for r in rows]
    require(actual == expected, 'wrong operator case coverage/order')
    for row in rows:
        require(row['result'] == 'PASS', 'operator result failed')
        require(math.isfinite(row['gpu_us']) and row['gpu_us'] > 0, 'invalid GPU interval')
        require(math.isfinite(row['fp64_max_abs']) and row['fp64_max_abs'] >= 0,
                'invalid FP64 error')
        require(all(row[key] == 0 for key in ('nonfinite', 'padding_changes',
                    'baseline_bit_mismatches', 'fp64_misses')), 'operator correctness failed')
    summary = []
    for case in CASES:
        times = {mode: [r['gpu_us'] for r in rows if
                       (r['n0'], r['n1'], r['q_scale'], r['mode']) == (*case, mode)]
                 for mode in MODES}
        base = statistics.median(times['baseline'])
        for mode in MODES:
            values = times[mode]
            median = statistics.median(values)
            summary.append(dict(n0=case[0], n1=case[1], q_scale=case[2], mode=mode,
                gpu_us=dict(median=median, min=min(values), max=max(values), cells=len(values)),
                speedup=base/median))
    return summary


def validate_resources(controller, rows):
    require(controller['result'] == 'PASS' and controller['error'] is None and
            controller['identities_unchanged'] is True, 'failed controller')
    require(controller['stop_returncode'] == controller['observer_returncode'] == 0, 'unclean stop')
    require(controller['full_model'] is False and controller['default_promotion'] is False,
            'operator scope changed')
    require(controller['available_before_bytes'] >= 12*2**30, 'operator admission failed')
    command = controller['command']
    require(command[:3] == ['systemd-run', '--user', '--service-type=exec'], 'uncontrolled launch')
    for setting in ('MemoryHigh=1G', 'MemoryMax=2G', 'MemorySwapMax=0',
                    'NoNewPrivileges=yes', 'LimitCORE=0', 'RuntimeMaxSec=600', 'RemainAfterExit=yes'):
        require(command.count(setting) == 1 and command[command.index(setting)-1] == '-p',
                'operator protection changed: '+setting)
    require(command[-1] == '4', 'wrong dispatch repetitions')
    require(len(rows) > 1 and 0 <= rows[0]['elapsed'] < 15, 'missing observations')
    gaps = [b['elapsed']-a['elapsed'] for a, b in zip(rows, rows[1:])]
    require(all(math.isfinite(gap) and 0 < gap <= 15 for gap in gaps), 'observation gap')
    require(rows[-1]['properties']['SubState'] in ('dead', 'exited') and
            rows[-1]['properties']['ExecMainStatus'] == '0', 'missing observed clean exit')
    final = controller['final_unit']
    require(final['SubState'] == 'exited' and final['ExecMainStatus'] == '0', 'operator did not exit cleanly')
    require(int(final['MemoryPeak']) <= 2*2**30 and final['MemorySwapPeak'] == '0', 'final memory cap/swap')
    pids, temperatures, peaks, events = set(), [], [], {}
    halo = False
    for row in rows:
        require(row['unit'] == controller['unit'] and row['violation'] is None, 'watchdog alarm/unit mismatch')
        require(row['memory']['MemAvailable'] >= 8*2**30, 'available memory below live floor')
        temps = [int(value) for value in row['temperatures'].values()]
        require(temps and all(0 < value < 93000 for value in temps), 'unsafe/missing temperatures')
        temperatures.extend(temps)
        props = row['properties']
        if int(props['MainPID']):
            pids.add(props['MainPID'])
            peaks.append(int(props['MemoryPeak']))
            require(props['MemorySwapPeak'] == '0', 'operator swapped')
            require({'max', 'oom', 'oom_kill'} <= row['events'].keys(), 'missing memory events')
        for key, value in row['events'].items():
            events[key] = max(events.get(key, 0), int(value))
        for device in row['devices'].values():
            compute = any(int(v.split()[0]) for k, v in device.items() if k.startswith('drm-engine-'))
            if device['drm-pdev'] == '0000:c1:00.0':
                halo |= compute
            else:
                require(not compute, 'non-Halo compute')
                require(memory_bytes(device['drm-memory-vram']) <= 2**20 and
                        memory_bytes(device['drm-memory-gtt']) <= 8*2**20, 'external GPU allocation')
    require(len(pids) == 1 and halo and peaks, 'missing/stale GPU process observation')
    require(max(peaks) <= 2*2**30 and all(events.get(k, 0) == 0 for k in
            ('max', 'oom', 'oom_kill', 'oom_group_kill')), 'hard memory limit/OOM event')
    return dict(samples=len(rows), maximum_gap_seconds=max(gaps),
                minimum_available_gib=min(r['memory']['MemAvailable'] for r in rows)/2**30,
                maximum_temperature_c=max(temperatures)/1000,
                memory_peak_bytes=max(max(peaks), int(final['MemoryPeak'])), swap_peak_bytes=0,
                memory_events=events)


def audit(folder, verify_files=True):
    folder = Path(folder)
    controller = json.loads((folder/'controller.json').read_text())
    log = (folder/'operator.log').read_text()
    rows = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
    require(rows == controller['rows'], 'controller/log rows differ')
    summary = validate_cells(rows)
    observations = [json.loads(line) for line in (folder/'observer.jsonl').read_text().splitlines()]
    resources = validate_resources(controller, observations)
    require('staged=yes; repetitions=4; synthetic only' in log, 'wrong placement/timing tier')
    if verify_files:
        for manifest in ('sources', 'artifacts'):
            require(bool(controller[manifest]), 'missing '+manifest)
            for path, expected in controller[manifest].items():
                require(sha(path) == expected, 'changed file: '+path)
    return dict(result='PASS', cells=len(rows), summary=summary, resources=resources,
        provenance={name:sha(folder/name) for name in ('controller.json', 'operator.log', 'observer.jsonl')},
        live_file_identity_verified=verify_files, full_model=False, default_promotion=False,
        caveat='Synthetic operator only; uploads and transposed-cache construction/maintenance excluded. '
               'Two order-balanced cells per mode/case, four timed dispatches per cell. '
               'No API, routing or full-model quality claim; cgroups omit some GPU allocations.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--archived', action='store_true', help='Do not rehash machine-local build files')
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, not args.archived), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
