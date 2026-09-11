#!/usr/bin/env python3
"""Per-stage decode wall-clock timing on the native dual-GPU split (dedicated GPU window).

Loads Halo 0:37 and XTX 37:48 through libqk, runs a 64-token batched prefill of
distinct ids per round, then 64 greedy decode steps timed per stage. Environment:
QK_GGUF (model), PRE_SLEEP (seconds to wait for the PLE prefetch), ROUND_LABELS,
ROUND_HOOK (shell command run after each round, {r} = finished round index).
"""
import ctypes as C, os, time, json, statistics, sys, subprocess
class Config(C.Structure):
    _fields_ = [("slots", C.c_uint32), ("ctx", C.c_uint32), ("chunk", C.c_uint32)]
model=os.environ["QK_GGUF"]
os.environ.update(QK_NATIVE_FLASH="1", QK_SHADER_DIR=os.path.abspath("build-halo/shaders"))
for k in ("QK_DEVICE_PCI","QK_DEVICE","QK_LAYER_DUMP"): os.environ.pop(k,None)
lib=C.CDLL(os.path.abspath("build-halo/libqk.so"))
u32,f32=C.c_uint32,C.c_float
lib.qk_open.argtypes=[C.c_char_p,C.POINTER(Config),C.c_void_p,C.c_size_t]; lib.qk_open.restype=C.c_void_p
lib.qk_close.argtypes=[C.c_void_p]
lib.qk_stage_run.argtypes=[C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32),u32,u32,C.POINTER(f32),C.POINTER(u32)]
engines=[]
for device,layers in [("STRIX_HALO","0:37"),("NAVI31","37:48")]:
    os.environ.update(QK_DEVICE_NAME=device,QK_LAYERS=layers)
    err=C.create_string_buffer(512)
    e=lib.qk_open(os.fsencode(model),C.byref(Config(1,4096,1)),err,len(err))
    if not e: raise RuntimeError(err.value.decode())
    engines.append(e)
time.sleep(float(os.environ.get("PRE_SLEEP","0")))
hidden=(f32*10240)(); ids=(u32*1)()
labels=os.environ.get("ROUND_LABELS","").split(",")
rounds=int(sys.argv[1]) if len(sys.argv)>1 else 3
for r in range(rounds):
    base=1000+r*5000
    prompt=(u32*64)(*range(base,base+64))
    h64=(f32*(10240*64))(); ids64=(u32*64)()
    t0=time.monotonic(); assert lib.qk_stage_run(engines[0],0,prompt,None,64,0,h64,None)==0; t1=time.monotonic()
    assert lib.qk_stage_run(engines[1],0,None,h64,64,0,None,ids64)==0; t2=time.monotonic()
    halo=[];xtx=[];tok=ids64[63]
    for step in range(64):
        t=(u32*1)(tok)
        a=time.monotonic(); assert lib.qk_stage_run(engines[0],0,t,None,1,64+step,hidden,None)==0
        b=time.monotonic(); assert lib.qk_stage_run(engines[1],0,None,hidden,1,64+step,None,ids)==0
        c=time.monotonic(); halo.append(b-a); xtx.append(c-b); tok=ids[0]
    def ms(v): return round(1000*statistics.median(v),2)
    print(json.dumps({"round":r,"label":labels[r] if r<len(labels) else "","prefill64_halo_s":round(t1-t0,3),"prefill64_xtx_s":round(t2-t1,3),"halo_stage_ms_median":ms(halo[8:]),"halo_min":round(1000*min(halo),2),"xtx_stage_ms_median":ms(xtx[8:]),"xtx_min":round(1000*min(xtx),2),"per_token_ms":ms([a+b for a,b in zip(halo[8:],xtx[8:])]),"tok_s":round(1000/ms([a+b for a,b in zip(halo[8:],xtx[8:])]),2)}),flush=True)
    hook=os.environ.get("ROUND_HOOK")
    if hook and r+1<rounds:
        subprocess.run(hook.replace("{r}",str(r)),shell=True)
        time.sleep(2)
for e in reversed(engines): lib.qk_close(e)
