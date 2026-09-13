#!/usr/bin/env python3
"""Read-only telemetry plus fail-closed stop of one explicitly named test unit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

unit, output = sys.argv[1:]
assert unit == 'qk-native-campaign-expert-split-jU4yTC'
halo = Path('/sys/bus/pci/devices/0000:c1:00.0')
def memory_bytes(value):
    fields = value.split()
    scale = {'B':1, 'KiB':1024, 'MiB':2**20, 'GiB':2**30}
    if len(fields) != 2 or fields[1] not in scale:
        raise ValueError('unexpected DRM memory units')
    return int(fields[0]) * scale[fields[1]]

start = time.monotonic()
with Path(output).open('x', buffering=1) as stream:
    while time.monotonic() - start < 7200:
        result = subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','SubState',
                  '-p','ExecMainStatus','-p','MemoryCurrent','-p','MemoryPeak','-p','MemorySwapCurrent',
                  '-p','MemorySwapPeak','-p','ControlGroup'], text=True)
        props = dict(line.split('=',1) for line in result.splitlines() if '=' in line)
        mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith(('MemAvailable:','AnonPages:','SwapFree:'))}
        pid = int(props.get('MainPID','0'))
        devices = {}
        violation = None
        if pid:
            for f in Path(f'/proc/{pid}/fdinfo').glob('*'):
                try:
                    fields = {k: v.strip() for line in f.read_text().splitlines() if line.startswith('drm-')
                              for k,v in [line.split(':',1)]}
                except FileNotFoundError:
                    continue
                if 'drm-pdev' not in fields:
                    continue
                devices[fields.get('drm-client-id',f.name)] = fields
                if fields['drm-pdev'] != '0000:c1:00.0':
                    if any(int(v.split()[0]) for k,v in fields.items() if k.startswith('drm-engine-')):
                        violation = 'non-Halo compute'
                    for key, limit in [('drm-memory-vram',2**20),('drm-memory-gtt',8*2**20)]:
                        if key in fields and memory_bytes(fields[key]) > limit:
                            violation = 'external GPU model allocation'
        telemetry = {f.name: f.read_text().strip() for f in halo.glob('hwmon/hwmon*/temp*_input')}
        if any(int(v) >= 93000 for v in telemetry.values()):
            violation = 'GPU temperature >=93C'
        if mem['MemAvailable'] < 8*2**30:
            violation = 'available memory below8GiB'
        cgroup = Path('/sys/fs/cgroup') / props.get('ControlGroup','').lstrip('/')
        events = {}
        if props.get('ControlGroup') and (cgroup/'memory.events').exists():
            events = dict(line.split() for line in (cgroup/'memory.events').read_text().splitlines())
            if any(int(events.get(k,'0')) for k in ('max','oom','oom_kill')):
                violation = 'hard memory limit/OOM event'
        row = {'elapsed':time.monotonic()-start,'unit':unit,'properties':props,'memory':mem,
               'devices':devices,'temperatures':telemetry,'events':events,'violation':violation}
        stream.write(json.dumps(row)+'\n')
        if violation:
            subprocess.run(['systemctl','--user','stop',unit],check=True,timeout=45)
            raise SystemExit(violation)
        if props.get('SubState') not in ('running','start','start-pre'):
            print(json.dumps({'observed_exit':props}),flush=True)
            break
        time.sleep(3)
    else:
        raise SystemExit('observer deadline exceeded')
