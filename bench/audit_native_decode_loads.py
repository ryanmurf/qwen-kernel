#!/usr/bin/env python3
"""Fail-closed checks for the same-order decode-load campaign."""
import math
import statistics

from audit_native_last_head_full import require
from audit_native_last_head_resources import memory_bytes

SIZES=(128,512,2048,8192,16384,31744)
ORDER=('serial','loads','loads','serial')


def validate_gate(reference, rows):
    expected={r['prompt_rows']:r['logit_sha256'] for r in reference
              if r['type']=='case_pass' and r['iteration']==0}
    require(len(expected)==15,'incomplete historical gate')
    require(rows[0]['type']=='metadata' and rows[-1]['type']=='closed','incomplete gate lifecycle')
    require(rows[0]['environment']['QK_ATTN_DECODE']=='loads','wrong gate decode mode')
    observed=[r for r in rows if r['type']=='row']
    wanted=[(n,it,i,h) for n,hashes in expected.items() for it in range(4) for i,h in enumerate(hashes)]
    require([(r['prompt_rows'],r['iteration'],r['row'],r['logit_sha256']) for r in observed]==wanted,
            'gate coverage/order or exact logits differ')
    require(len(observed)==664 and all(0<=r['greedy']<248320 for r in observed),'invalid gate rows')
    cases=[r for r in rows if r['type']=='case_pass']
    require([(r['prompt_rows'],r['iteration'],r['logit_sha256']) for r in cases]==
            [(n,it,h) for n,h in expected.items() for it in range(4)],'gate case summaries differ')
    results=[r for r in rows if r['type']=='result']
    require(len(results)==1 and results[0]==dict(type='result',result='PASS',cases=15,compared_rows=664,
            bit_exact=True,identities_unchanged=True,full_model=True),'gate did not pass completely')
    return dict(result='PASS',rows=664,cases=15,bit_exact=True)


def validate_resources(rows, unit, final, model=True):
    require(len(rows)>=2 and 0<=rows[0]['elapsed']<15,'missing/late observer')
    require(rows[-1]['properties']['SubState'] in ('dead','exited') and
            rows[-1]['properties']['ExecMainStatus']=='0','no clean observed shutdown')
    gaps=[b['elapsed']-a['elapsed'] for a,b in zip(rows,rows[1:])]
    require(all(math.isfinite(g) and 0<g<=15 for g in gaps),'resource coverage gap')
    peaks=[int(final['MemoryPeak'])]
    swaps=[int(final['MemorySwapPeak'])]
    temperatures=[]
    observed_halo=False
    for row in rows:
        require(row['unit']==unit and row['violation'] is None,'wrong unit or watchdog alarm')
        require(row['memory']['MemAvailable']>=8*2**30,'memory floor breached')
        require(row['temperatures'] and all(0<int(t)<93000 for t in row['temperatures'].values()),'temperature invalid')
        temperatures.extend(int(t) for t in row['temperatures'].values())
        props=row['properties']
        if int(props['MainPID']):
            require({'max','oom','oom_kill'} <= row['events'].keys(),'missing live cgroup events')
            peaks.append(int(props['MemoryPeak']))
            swaps.append(int(props['MemorySwapPeak']))
        require(all(int(row['events'].get(k,0))==0 for k in ('max','oom','oom_kill')),'memory-max/OOM event')
        for device in row['devices'].values():
            engines=[int(v.split()[0]) for k,v in device.items() if k.startswith('drm-engine-')]
            if device['drm-pdev']=='0000:c1:00.0':
                observed_halo |= any(v>0 for v in engines)
            else:
                require(all(v==0 for v in engines),'non-Halo engine work')
                for name,limit in (('vram',2**20),('gtt',8*2**20)):
                    require(memory_bytes(device['drm-memory-'+name])<=limit,'external GPU allocation')
    require(observed_halo,'missing observed Halo work')
    require(max(peaks)<=(32 if model else 2)*2**30 and max(swaps)<=(512*2**20 if model else 0),
            'cgroup budget exceeded')
    return dict(result='PASS',samples=len(rows),max_gap_seconds=max(gaps),
                minimum_available_gib=min(r['memory']['MemAvailable'] for r in rows)/2**30,
                maximum_temperature_c=max(temperatures)/1000,memory_peak_bytes=max(peaks),swap_peak_bytes=max(swaps))


def compare(runs):
    require(len(runs)==4,'four independent launches required')
    cells={}
    reference_hashes={}
    for index,(mode,rows) in enumerate(zip(ORDER,runs)):
        require(rows[0]['type']=='metadata' and rows[0]['mode']==mode and rows[0]['launch']==index,'launch order')
        require(rows[0]['context']==32768 and rows[0]['sizes']==list(SIZES),'context/size differs')
        require(rows[-1]==dict(type='result',result='PASS',cases=6),'matrix incomplete')
        if index:
            for key in ('fixture_sha256','prompt_sha256','harness_sha256','protocol'):
                require(rows[0][key]==runs[0][0][key],f'matrix identity differs: {key}')
        measured=[r for r in rows if r['type']=='decode']
        require([r['prompt_tokens'] for r in measured]==list(SIZES),'missing/reordered context size')
        for row in measured:
            n=row['prompt_tokens']
            count=512 if n>=16384 else 128
            require(row['exact_output_length'] is True and row['coherent_counting_prefix'] is True and
                    row['streamed_tokens']==count and row['requested_output_tokens']==count,'invalid counting output')
            hashes=(row['output_token_sha256'],row['output_text_sha256'])
            require(all(isinstance(h,str) and len(h)==64 for h in hashes),'missing output hashes')
            if not index: reference_hashes[n]=hashes
            require(hashes==reference_hashes[n],'cross-mode output differs')
            for key in ('ttft_seconds','decode_tokens_per_second','stream_total_seconds'):
                require(type(row[key]) in (int,float) and math.isfinite(row[key]) and row[key]>0,'invalid timing')
            cells.setdefault((mode,n),[]).append(row)
    summary=[]
    ratios={}
    keys=('ttft_seconds','decode_tokens_per_second','stream_total_seconds')
    for n in SIZES:
        medians={}
        for mode in ('serial','loads'):
            values={k:[r[k] for r in cells[mode,n]] for k in keys}
            medians[mode]={k:statistics.median(v) for k,v in values.items()}
            summary.append(dict(mode=mode,prompt_tokens=n,output_tokens=512 if n>=16384 else 128,
                                **{k:dict(median=statistics.median(v),min=min(v),max=max(v),n=len(v)) for k,v in values.items()}))
        ratios[n]={k:medians['loads'][k]/medians['serial'][k] for k in keys}
    promote=(all(ratios[n]['decode_tokens_per_second']>=1.25 for n in (16384,31744)) and
             all(r['decode_tokens_per_second']>=.97 and r['stream_total_seconds']<=1.03 and
                 r['ttft_seconds']<=1.03 for r in ratios.values()))
    return dict(result='PASS',summary=summary,ratios=ratios,loads_promoted=promote,
                criteria='16K/31K median decode >=25% faster; no tested median decode loss >3%, total or TTFT increase >3%',
                caveat='Two independent launches per mode; fresh KV, not cold file cache; counting is not a quality evaluation.')
