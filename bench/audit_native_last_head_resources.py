#!/usr/bin/env python3
"""Audit the completed last-head HTTP controller and its sampled resource log.

This checks observations, not continuous safety or whole-host GPU exclusivity.
Cgroup memory counters do not account for all GPU allocations on this node.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_native_last_head_full import require

GIB = 2**30
HALO = '0000:c1:00.0'


def memory_bytes(value):
    fields = value.split()
    require(len(fields) in (1, 2), 'invalid DRM memory field')
    scale = {'B': 1, 'KiB': 1024, 'MiB': 2**20, 'GiB': GIB}
    require(len(fields) == 1 or fields[1] in scale, 'unknown DRM memory unit')
    number = int(fields[0])
    require(number >= 0, 'negative DRM allocation')
    return number * (scale[fields[1]] if len(fields) == 2 else 1)


def validate_resources(controller, rows):
    mode = controller['mode']
    require(mode in ('all', 'last'), 'unknown trial mode')
    require(controller['unit'].startswith(f'qk-last-head-http-{mode}-'), 'wrong unit')
    require(controller['error'] is None and controller['identities_unchanged'] is True,
            'controller failed or build identity changed')
    for key in ('http', 'benchmark', 'audit', 'stop', 'observer'):
        require(controller[key + '_returncode'] == 0, f'{key} did not complete successfully')
    require(controller['default_promotion'] is False, 'trial changed the serving default')
    require(controller['actual_environment']['QK_FLASH_PREFILL_LAST'] ==
            ('1' if mode == 'last' else '0'), 'wrong observed prefill policy')
    admission = controller['admission']
    require(admission['admitted'] is True and admission['minimum_available_gib'] == 24
            and admission['maximum_other_memory_gib'] == 12
            and admission['available_kib'] >= 24 * 2**20
            and admission['anonymous_shared_and_swap_kib'] <= 12 * 2**20,
            'launch admission gate not preserved')
    final = controller['final_unit']
    require(final['SubState'] == 'running' and int(final['MainPID']) > 0
            and final['ExecMainStatus'] == '0', 'server exited before controlled stop')
    require(len(rows) >= 2, 'resource observations missing')
    require(0 <= rows[0]['elapsed'] < 15, 'resource observer started late')
    require(rows[-1]['properties']['SubState'] in ('dead', 'exited')
            and rows[-1]['properties']['ExecMainStatus'] == '0', 'no clean observed stop')
    gaps = [b['elapsed'] - a['elapsed'] for a, b in zip(rows, rows[1:])]
    require(all(math.isfinite(gap) and 0 < gap <= 15 for gap in gaps), 'resource coverage gap')
    available, temperatures, swap, peaks = [], [], [], []
    events, external = {}, {}
    observed_halo_compute = False
    for row in rows:
        require(row['unit'] == controller['unit'] and row['violation'] is None,
                'wrong observed unit or watchdog alarm')
        require(row['memory']['MemAvailable'] >= 8 * GIB, 'available RAM below safety floor')
        available.append(row['memory']['MemAvailable'])
        values = list(row['temperatures'].values())
        require(values and all(0 < int(v) < 93000 for v in values), 'invalid/unsafe GPU temperature')
        temperatures.extend(int(v) for v in values)
        props = row['properties']
        if int(props['MainPID']) > 0:
            require(props['MainPID'] == final['MainPID'], 'server PID changed')
            require(props['SubState'] in ('running', 'start', 'start-pre'), 'unexpected live server state')
            require({'max', 'oom', 'oom_kill'} <= row['events'].keys(), 'missing live memory events')
            swap.append(int(props['MemorySwapPeak']))
            peaks.append(int(props['MemoryPeak']))
        for key, value in row['events'].items():
            events[key] = max(events.get(key, 0), int(value))
        require(all(int(row['events'].get(key, 0)) == 0 for key in ('max', 'oom', 'oom_kill')),
                'hard memory limit or OOM event')
        for device in row['devices'].values():
            engines = [int(v.split()[0]) for k, v in device.items() if k.startswith('drm-engine-')]
            if device['drm-pdev'] == HALO:
                observed_halo_compute |= any(value > 0 for value in engines)
                continue
            require(all(value == 0 for value in engines), 'non-Halo engine work')
            summary = external.setdefault(device['drm-pdev'], {'vram_peak_bytes': 0, 'gtt_peak_bytes': 0})
            for name, limit in (('vram', 2**20), ('gtt', 8 * 2**20)):
                field = 'drm-memory-' + name
                require(field in device, 'missing external GPU memory accounting')
                allocation = memory_bytes(device[field])
                require(allocation <= limit, 'external GPU model-sized allocation')
                summary[name + '_peak_bytes'] = max(summary[name + '_peak_bytes'], allocation)
    require(observed_halo_compute and peaks and swap, 'missing live Halo/server observations')
    peak = max(int(final['MemoryPeak']), *peaks)
    swap_peak = max(int(final['MemorySwapPeak']), *swap)
    require(peak <= 32 * GIB and swap_peak <= 512 * 2**20, 'cgroup budget exceeded')
    return {'result': 'PASS', 'mode': mode, 'unit': controller['unit'],
            'start_utc': controller['start_utc'], 'end_utc': controller['end_utc'],
            'samples': len(rows), 'observed_seconds': rows[-1]['elapsed'],
            'maximum_sample_gap_seconds': max(gaps),
            'minimum_available_gib': min(available) / GIB,
            'maximum_gpu_temperature_c': max(temperatures) / 1000,
            'cgroup_memory_peak_bytes': peak,
            'cgroup_swap_peak_bytes': swap_peak,
            'memory_event_maxima': events, 'external_gpu_observations': external,
            'cleanup': controller['cleanup'], 'admission': admission,
            'identities_unchanged': True, 'controlled_stop': True,
            'caveat': 'Sampled server fdinfo only; cgroup memory excludes some GPU allocations.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controller', type=Path)
    parser.add_argument('observer', type=Path)
    args = parser.parse_args()
    controller = json.loads(args.controller.read_text())
    rows = [json.loads(line) for line in args.observer.read_text().splitlines()]
    result = validate_resources(controller, rows)
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (args.controller, args.observer)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
