#!/usr/bin/env python3
"""One exclusive, bounded expert layout trial; unchanged serving restored in finally."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/home/ryan/IdeaProjects/qwen-kernel-strix-halo')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/ryan/qk-combined-profile-kGqp5a')
import stage
sys.path.insert(0,str(ROOT/'bench'))
from audit_native_decode_loads import validate_resources
UNIT='qk-native-campaign-expert-split-jU4yTC'
require=stage.require
def log(**row): print(json.dumps(dict(utc=stage.now(),**row)),flush=True)
def run(argv,name,timeout=240,env=None):
    log(start=name)
    with (OUT/name).open('x') as stream:
        subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,check=True,timeout=timeout)
    log(complete=name)
def serving():
    props=stage.properties('qwen-native-flash-server32')
    require(props['SubState']=='running' and int(props['MainPID'])>0,'server not running')
    p=Path('/proc')/props['MainPID']
    require(os.readlink(p/'exe')==str(ROOT/'build-halo/rust/release/server'),'unexpected server binary')
    env=dict(s.split('=',1) for s in (p/'environ').read_text().split('\0') if s.startswith('QK_'))
    require(env.get('QK_ATTN_DECODE')=='serial' and env.get('QK_FLASH_ATTN_BATCH')=='vec4','unexpected policies')
    return dict(pid=props['MainPID'],environment=env,executable_sha256=stage.sha(p/'exe'))
def drained():
    end=time.monotonic()+90
    while True:
        try:
            stage.idle()
            for p in Path('/proc').glob('[0-9]*'):
                try: name=(p/'comm').read_text().strip()
                except (FileNotFoundError,PermissionError,ProcessLookupError): continue
                require(name!='operator','another GPU operator active')
            return
        except RuntimeError:
            require(time.monotonic()<end,'devices not drained')
            time.sleep(3)
def main():
    require(not (OUT/'result.json').exists() and not (OUT/'manifest.json').exists(),'already attempted')
    preceding=json.loads(Path('/home/ryan/qk-expert-layout-tAFAFc/result.json').read_text())
    require(preceding.get('result')=='PASS' and preceding.get('restored') is True,'previous expert trial incomplete')
    previous=json.loads(Path('/home/ryan/qk-decode-full-MFem1G/result.json').read_text())
    require(previous.get('restored') is True,'previous serving restoration incomplete')
    require(json.loads(Path('/home/ryan/qk-decode-full-MFem1G/restored.audit.json').read_text())['result']=='PASS',
            'previous restoration audit failed')
    installed={str(p):stage.sha(p) for p in [ROOT/'build-halo/libqk.so',ROOT/'build-halo/rust/release/server']+
               sorted((ROOT/'build-halo/shaders').glob('*.spv'))}
    require(installed[str(ROOT/'build-halo/libqk.so')]=='068369b87806608ed7d00efb2b72d3bd3e0f19abfd0ed67d7f452b8e952d8931',
            'fallback library changed')
    require(stage.sha(OUT/'baseline.spv')==stage.sha(ROOT/'build-halo/shaders/qwen4_moe_gateup_tiled.spv'),
            'baseline shader changed')
    for mode in ('baseline','soa','split'):
        subprocess.run(['spirv-val','--target-env','vulkan1.2',str(OUT/(mode+'.spv'))],check=True)
    files=[Path(__file__),OUT/'watch.py',OUT/'operator',ROOT/'tests/halo_expert_split.cpp',
           ROOT/'tests/halo_expert_split.comp',ROOT/'tests/halo_gemm_operator_common.h',
           Path('/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so')]+[OUT/(s+'.spv') for s in ('baseline','soa','split')]
    files+=sorted(p for p in (ROOT/'src').rglob('*') if p.is_file() and p.suffix in ('.h','.hpp','.cpp'))
    record=dict(start_utc=stage.now(),before=serving(),installed=installed,
                files={str(p):stage.sha(p) for p in files},kernel=os.uname().release,
                full_model=False,default_promotion=False)
    stage.save(OUT/'manifest.json',record)
    touched=False; started=False; observer=None
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('terminated')))
    try:
        touched=True
        subprocess.run(['systemctl','--user','stop','qwen-native-flash-router','qwen-native-flash-server32'],check=True,timeout=60)
        drained(); record['admission']=stage.memory(12)
        command=['systemd-run','--user','--service-type=exec',f'--unit={UNIT}',
            '-p','MemoryHigh=1G','-p','MemoryMax=2G','-p','MemorySwapMax=0','-p','NoNewPrivileges=yes',
            '-p','LimitCORE=0','-p','RuntimeMaxSec=300','-p','RemainAfterExit=yes',
            '-p',f'StandardOutput=append:{OUT}/operator.log','-p',f'StandardError=append:{OUT}/operator.stderr',
            '/usr/bin/env','-i','PATH=/usr/bin:/bin','LANG=C.UTF-8',f'QK_SHADER_DIR={OUT}',str(OUT/'operator'),str(stage.MODEL)]
        record['command']=command
        subprocess.run(command,check=True); started=True
        observer=subprocess.Popen([sys.executable,str(OUT/'watch.py'),UNIT,str(OUT/'observer.jsonl')])
        deadline=time.monotonic()+330
        while True:
            props=stage.properties(UNIT)
            if props['SubState'] not in ('running','start','start-pre'): break
            require(time.monotonic()<deadline and observer.poll() is None,'deadline/watchdog')
            time.sleep(3)
        record['final_unit']=props
        record['observer_returncode']=observer.wait(timeout=15)
        require(props['SubState']=='exited' and props['ExecMainStatus']=='0','operator failed')
        require(record['observer_returncode']==0,'observer failed')
        rows=[json.loads(s) for s in (OUT/'operator.log').read_text().splitlines() if s.startswith('{')]
        require(rows[-1]==dict(type='result',result='PASS',cells=72,full_model=False),'incomplete operator')
        cells=[r for r in rows if r['type']=='cell']
        cases=[(32,1,1),(32,8,1),(32,16,1),(32,17,1),(1,64,8),(16,3,.25)]
        expected=[(layer,a,n,scale,mode,i) for layer in (0,47) for a,n,scale in cases
                  for i,mode in enumerate(('baseline','soa','split','split','soa','baseline'))]
        require([(r['layer'],r['active_experts'],r['pairs_per_expert'],r['q_scale'],r['mode'],r['order'])
                 for r in cells]==expected,'coverage/order differs')
        require(all(r['result']=='PASS' and math.isfinite(r['gpu_us']) and r['gpu_us']>0 and
                    all(r[k]==0 for k in ('bit_mismatches','nonfinite','padding_changes','fp64_misses')) for r in cells),
                'numerical failure')
        record['resources']=validate_resources([json.loads(s) for s in (OUT/'observer.jsonl').read_text().splitlines()],
                                              UNIT,props,False)
        require(all(stage.sha(Path(p))==h for p,h in record['files'].items()),'trial inputs changed')
        record.update(result='PASS',error=None,log_sha256=stage.sha(OUT/'operator.log'))
    except BaseException as e:
        record.update(result='FAIL',error=f'{type(e).__name__}: {e}')
        log(error=record['error'])
    finally:
        if started:
            subprocess.run(['systemctl','--user','stop',UNIT],check=False,timeout=45)
        if observer is not None and observer.poll() is None:
            try: observer.wait(timeout=15)
            except subprocess.TimeoutExpired: observer.terminate(); observer.wait(timeout=15)
        if touched:
            try:
                require(all(stage.sha(Path(p))==h for p,h in installed.items()),'installed fallback changed')
                drained(); record['restore_cleanup']=stage.cleanup(); record['restore_admission']=stage.memory(24)
                env={**os.environ,'QK_FLASH_PREFILL_LAST':'1','QK_FLASH_ATTN_BATCH':'vec4','QK_FLASH_BATCH':'512',
                     'QK_FLASH_GEMM':'baseline','QK_ATTN_DECODE':'serial','QK_FLASH_COOPMAT':'0','QK_PLE_PREFETCH':'0'}
                run(['bash','deploy/restore-native-flash.sh',str(stage.MODEL),'32768','0','single'],'restore.log',420,env)
                for port in (8194,8091):
                    cmd=[sys.executable,'-E','tests/native_flash_http.py','--url',f'http://127.0.0.1:{port}']
                    if port==8091: cmd.append('--skip-cancel')
                    run(cmd,f'restored-http{port}.jsonl')
                run([sys.executable,'/home/ryan/qk-last-head-full-VjRrMt/proxy-smoke.py'],'restored-http8092.jsonl')
                record['after']=serving()
                require(record['before']['environment']==record['after']['environment'],'restore policies differ')
                require(record['before']['executable_sha256']==record['after']['executable_sha256'],'server differs')
                record['restored']=True
            except BaseException as e:
                record.update(restored=False,restore_error=f'{type(e).__name__}: {e}')
        stage.save(OUT/'result.json',record)
    log(result=record['result'],restored=record.get('restored'))
    require(record['result']=='PASS' and record.get('restored') is True,'trial/restoration did not pass')
if __name__=='__main__': main()
