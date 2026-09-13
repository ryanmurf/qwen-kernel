#!/usr/bin/env python3
"""Bounded CPU-only microbenchmark. Serving remains unchanged; reject GPU overlap."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

OUT=Path(__file__).resolve().parent
ROOT=Path('/home/ryan/IdeaProjects/qwen-kernel-strix-halo')
MODEL=Path('/home/ryan/models/Qwen3.8-Flash-Next-Uncensored-Q5_K_M/Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf')
UNIT='qk-cpu-prep-repeat-YOjQvm'
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def props(unit):
    return dict(line.split('=',1) for line in subprocess.check_output(
        ['systemctl','--user','show',unit,'-p','MainPID','-p','SubState','-p','ExecMainStatus',
         '-p','MemoryPeak','-p','MemorySwapPeak','-p','CPUUsageNSec'],text=True).splitlines())
def serving():
    p=props('qwen-native-flash-server32')
    assert p['SubState']=='running' and int(p['MainPID'])>0
    engines={}
    for file in (Path('/proc')/p['MainPID']/'fdinfo').glob('*'):
        s=file.read_text()
        if 'drm-client-id:' not in s: continue
        fields=dict(line.split(':',1) for line in s.splitlines() if line.startswith('drm-'))
        engines[fields['drm-client-id'].strip()]={k:v.strip() for k,v in fields.items() if k.startswith('drm-engine-')}
    return {'pid':p['MainPID'],'engines':engines}
assert not (OUT/'result.json').exists() and not (OUT/'raw.jsonl').exists()
files=[OUT/'bench',OUT/'tokens.txt',Path(__file__),ROOT/'tests/halo_cpu_prep.cpp',
       ROOT/'tests/halo_cpu_prep_simd.h',ROOT/'src/quants.h',ROOT/'src/qwen4_ple.h']
record={'sources':{str(p):sha(p) for p in files},'installed_before':{
    str(p):sha(p) for p in (ROOT/'build-halo/libqk.so',ROOT/'build-halo/rust/release/server')},
    'serving_before':serving(),'gpu_used':False,'production_changed':False,
    'protocol':'CPU-only, core 2 pinned; no file-cache flush; ABBAABBA warm preparation, page lookup separate'}
command=['systemd-run','--user','--service-type=exec',f'--unit={UNIT}',
    '-p','MemoryHigh=768M','-p','MemoryMax=1G','-p','MemorySwapMax=0','-p','NoNewPrivileges=yes',
    '-p','LimitCORE=0','-p','RuntimeMaxSec=120','-p','RemainAfterExit=yes','-p','CPUAffinity=2',
    '-p',f'StandardOutput=append:{OUT}/raw.jsonl','-p',f'StandardError=append:{OUT}/stderr.log',
    str(OUT/'bench'),str(MODEL),str(OUT/'tokens.txt')]
record['command']=command
try:
    start=time.monotonic()
    subprocess.run(command,check=True,timeout=15)
    while props(UNIT)['SubState'] in ('running','start','start-pre'):
        assert time.monotonic()-start<135, 'CPU test timeout'
        time.sleep(.1)
    assert props(UNIT)['SubState']=='exited' and props(UNIT)['ExecMainStatus']=='0', 'CPU worker failed'
    record.update(elapsed_seconds=time.monotonic()-start,final_unit=props(UNIT),serving_after=serving())
    rows=[json.loads(s) for s in (OUT/'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['result']=='PASS'
    assert record['final_unit']['MemorySwapPeak']=='0'
    assert record['serving_before']==record['serving_after'], 'GPU serving overlapped; timing not clean'
    assert all(sha(Path(p))==h for p,h in record['sources'].items())
    assert all(sha(Path(p))==h for p,h in record['installed_before'].items())
    record.update(result='PASS',raw_sha256=sha(OUT/'raw.jsonl'))
except BaseException as e:
    record.update(result='FAIL',error=str(e))
finally:
    subprocess.run(['systemctl','--user','stop',UNIT],check=False,timeout=15)
    with (OUT/'result.json').open('x') as f: json.dump(record,f,indent=2); f.write('\n')
print(json.dumps(record,indent=2))
assert record['result']=='PASS'
