#!/usr/bin/env python3
"""Offline combined-prefill gate, instrumented profile and HTTP campaign audits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import audit_native_last_head_full as old
import native_last_head_full as full
from audit_native_last_head_resources import memory_bytes
import prefill_matrix as pm

ORDER = ['all','last','vec4','combined','combined','vec4','last','all']
require = old.require


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_combined(rows, reference):
    old.validate_gate(reference)
    require(rows[0]['type']=='metadata' and rows[-1]['type']=='closed','combined gate not closed')
    meta = rows[0]
    require(meta['experiment']=='native-combined-gate-v1' and meta['context']==32768 and meta['chunk']==512,
            'wrong combined gate configuration')
    require(meta['environment']=={**full.ENV,'QK_FLASH_ATTN_BATCH':'vec4',
                                 'QK_SHADER_DIR':meta['environment'].get('QK_SHADER_DIR')},'wrong combined environment')
    for key, refkey in [('library','library_sha256'),('model','model'),('shaders','shaders')]:
        require(meta['identity'][key]==reference[0][refkey],'different reference '+key)
    expected = [r for r in reference if r['type'] in ('row','case_pass')]
    actual = [r for r in rows if r['type'] in ('row','case_pass')]
    require(len(actual)==len(expected),'missing combined gate records')
    hashes = {r['prompt_rows']:r['logit_sha256'] for r in reference if r['type']=='case_pass' and r['iteration']==0}
    for a,b in zip(actual,expected):
        for key in ('type','prompt_rows','iteration','last_only'):
            require(a[key]==b[key],'reordered combined records')
        if a['type']=='row':
            for key in ('row','base','width','greedy'):
                require(a[key]==b[key],'combined row differs')
            require(a['logit_sha256']==hashes[a['prompt_rows']][a['row']],'combined logits differ')
            require(pm.positive_number(a['seconds']),'invalid gate timing')
        else:
            require(a['logit_sha256']==b['logit_sha256'],'combined case logits differ')
    results = [r for r in rows if r['type']=='result']
    require(len(results)==1 and results[0]==dict(type='result',result='PASS',cases=15,compared_rows=664,
            bit_exact=True,identities_unchanged=True,full_model=True),'missing combined result')
    return dict(result='PASS',cases=15,compared_full_vocabulary_rows=664,unique_reference_rows=166,
                bit_exact=True,reference_kind='same-library frozen baseline attention')


def category(shader):
    if shader.startswith('fa_'):
        return 'attention'
    if 'moe' in shader:
        return 'experts_and_routing'
    if 'gdn' in shader or shader.startswith('dn_'):
        return 'recurrent'
    if 'hc.' in shader:
        return 'hyperconnections'
    if 'ple' in shader:
        return 'ple_gpu'
    if 'gemm' in shader or 'gemv' in shader or 'argmax' in shader:
        return 'dense_projections_and_head'
    return 'other'


def validate_profile(rows, log):
    require(rows[0]['type']=='metadata' and rows[-1]['type']=='closed','profile not closed')
    meta = rows[0]
    require(meta['experiment']=='native-long-profile-v1' and meta['context']==32768 and meta['chunk']==512,
            'wrong profile configuration')
    require(meta['environment']=={**full.ENV,'QK_FLASH_PROFILE':'1','QK_FLASH_REPLAY':'0',
            'QK_FLASH_TIMING':'1','QK_SHADER_DIR':meta['environment'].get('QK_SHADER_DIR')},'wrong profile environment')
    expected = [(phase,size,index,base,width) for size in (2048,8192,16384)
                for phase,index,base,width in
                ([('prefill',i,b,512) for i,b in enumerate(range(0,size,512))] +
                 [('decode',i,size+i,1) for i in range(16)])]
    steps = [r for r in rows if r['type']=='step']
    require([tuple(r[k] for k in ('phase','prompt_tokens','index','base','width')) for r in steps]==expected,
            'missing/reordered profile steps')
    blocks = []
    active = None
    for line in log.splitlines():
        if line.startswith('[campaign begin] '):
            require(active is None,'nested profile marker')
            active = {'marker':json.loads(line[len('[campaign begin] '):]),'shaders':{}}
        elif line.startswith('[campaign end] '):
            require(active is not None and active['marker']==json.loads(line[len('[campaign end] '):]),
                    'unmatched profile marker')
            require('gpu_ms' in active and active['shaders'],'missing GPU profile')
            blocks.append(active)
            active = None
        elif active is not None:
            match = re.match(r'\[flash profile\] (batched forward, (\d+) tokens|serial forward, 1 token): ([0-9.]+) ms GPU',line)
            if match:
                require('gpu_ms' not in active,'duplicate GPU block')
                require(int(match[2] or 1)==active['marker']['width'],'wrong GPU block width')
                active['gpu_ms'] = float(match[3])
            match = re.match(r'\[flash profile\]\s+(\S+)\s+([0-9.]+) ms\s+[0-9.]+% \((\d+) dispatches',line)
            if match:
                require(match[1] not in active['shaders'],'duplicate shader attribution')
                active['shaders'][match[1]] = {'ms':float(match[2]),'dispatches':int(match[3])}
    require(active is None and len(blocks)==len(steps),'incomplete profile attribution')
    summary = []
    for step,block in zip(steps,blocks):
        require(all(step[k]==v for k,v in block['marker'].items()),'profile marker/step mismatch')
        require(pm.positive_number(step['seconds']) and pm.positive_number(block['gpu_ms']),'invalid profile timing')
        require(abs(sum(v['ms'] for v in block['shaders'].values())-block['gpu_ms'])<.05,'GPU totals disagree')
    for size in (2048,8192,16384):
        for phase in ('prefill','decode'):
            selected = [(s,b) for s,b in zip(steps,blocks) if s['prompt_tokens']==size and s['phase']==phase]
            divisor = 16 if phase=='decode' else 1
            gpu = sum(b['gpu_ms'] for s,b in selected)/divisor
            shaders = {}
            groups = {}
            for step,block in selected:
                for name,values in block['shaders'].items():
                    row = shaders.setdefault(name,{'ms':0,'dispatches':0})
                    row['ms'] += values['ms']/divisor
                    row['dispatches'] += values['dispatches']/divisor
                    key = category(name)
                    groups[key] = groups.get(key,0)+values['ms']/divisor
            summary.append(dict(prompt_tokens=size,phase=phase,
                units='ms per decode token' if phase=='decode' else 'ms for whole prefill',
                gpu_ms=gpu,wall_ms=sum(s['seconds'] for s,b in selected)*1000/divisor,
                category_ms=groups,category_percent={k:v/gpu*100 for k,v in groups.items()},
                shaders=dict(sorted(shaders.items(),key=lambda x:-x[1]['ms']))))
    require([r['prompt_tokens'] for r in rows if r['type']=='case_pass']==[2048,8192,16384], 'missing profile cases')
    require([r for r in rows if r['type']=='result']==[dict(type='result',result='PASS',cases=3)],'missing profile result')
    return dict(result='PASS',steps=len(steps),summary=summary,
                caveat='Instrumented fenced GPU intervals, not uninstrumented API throughput; category labels group shader names.')


def validate_resources(controller, rows, stage=False):
    require(controller['error'] is None and controller['identities_unchanged'] is True,'controller failed')
    require(controller['stop_returncode']==controller['observer_returncode']==0,'unclean stop/observer')
    admission = controller['admission']
    require(admission['admitted'] and admission['minimum_available_gib']==24 and admission['maximum_other_memory_gib']==12
            and admission['available_kib']>=24*2**20 and admission['anonymous_shared_and_swap_kib']<=12*2**20,'admission changed')
    require(len(rows)>1 and 0<=rows[0]['elapsed']<15,'missing resource coverage')
    gaps = [b['elapsed']-a['elapsed'] for a,b in zip(rows,rows[1:])]
    require(all(math.isfinite(x) and 0<x<=15 for x in gaps),'resource sample gap')
    require(rows[-1]['properties']['SubState'] in ('dead','exited') and
            rows[-1]['properties']['ExecMainStatus']=='0','missing clean observed exit')
    pids = set()
    temperatures, peaks, swaps = [], [], []
    events = {}
    halo = False
    for row in rows:
        require(row['unit']==controller['unit'] and row['violation'] is None,'resource alarm/wrong unit')
        require(row['memory']['MemAvailable']>=8*2**30,'low available memory')
        temps = list(map(int,row['temperatures'].values()))
        require(temps and all(0<x<93000 for x in temps),'unsafe/missing temperatures')
        temperatures.extend(temps)
        p = row['properties']
        if int(p['MainPID']):
            pids.add(p['MainPID'])
            peaks.append(int(p['MemoryPeak']))
            swaps.append(int(p['MemorySwapPeak']))
            require({'max','oom','oom_kill'}<=row['events'].keys(),'missing memory events')
        for k,v in row['events'].items():
            events[k]=max(events.get(k,0),int(v))
        for device in row['devices'].values():
            compute = any(int(v.split()[0]) for k,v in device.items() if k.startswith('drm-engine-'))
            if device['drm-pdev']=='0000:c1:00.0':
                halo |= compute
            else:
                require(not compute,'external GPU compute')
                require(memory_bytes(device['drm-memory-vram'])<=2**20 and
                        memory_bytes(device['drm-memory-gtt'])<=8*2**20,'external GPU weights')
    require(len(pids)==1 and halo and peaks and swaps,'missing/stale process observations')
    require(max(peaks)<=32*2**30 and max(swaps)<=512*2**20,'cgroup cap exceeded')
    require(all(events.get(k,0)==0 for k in ('max','oom','oom_kill')),'hard memory events')
    if not stage:
        require(controller['final_unit']['MainPID'] in pids and controller['final_unit']['SubState']=='running',
                'server not live at controlled stop')
        for name in ('http','short_benchmark','long_benchmark','short_audit','long_audit','code'):
            require(controller[name+'_returncode']==0,'missing completed '+name)
    return dict(result='PASS',samples=len(rows),maximum_gap_seconds=max(gaps),
                minimum_available_gib=min(r['memory']['MemAvailable'] for r in rows)/2**30,
                maximum_temperature_c=max(temperatures)/1000,memory_peak_bytes=max(peaks),
                swap_peak_bytes=max(swaps),memory_events=events,
                caveat='Sampled model-process DRM only; cgroups exclude some GPU allocations.')


def stats(values):
    return {'median':statistics.median(values),'min':min(values),'max':max(values),'n':len(values)}


def validate_http(folder):
    folder = Path(folder)
    validate_combined(read(folder/'gate.jsonl'),read('/home/ryan/qk-last-head-full-VjRrMt/gate.jsonl'))
    expected_build = json.loads((folder/'http-build.json').read_text())
    fixture = pm.load_fixture('/home/ryan/qk-prefill-matrix-eyUgzZ/shared-fixture.json')
    compared = {}
    seeded = set()
    code = {}
    code_metadata = None
    code_measurements = []
    measurements = []
    resources = []
    for index,mode in enumerate(ORDER):
        prefix = folder/f'http-{index:02d}-{mode}'
        ctrl = json.loads(Path(str(prefix)+'.controller.json').read_text())
        require(ctrl['launch_index']==index and ctrl['mode']==mode and ctrl['order']==ORDER,'wrong launch order')
        resources.append(validate_resources(ctrl,read(str(prefix)+'.observer.jsonl')))
        env = {**full.ENV,'QK_SHADER_DIR':str(Path('/home/ryan/qk-last-head-IIfPmw/build/shaders')),
               'QK_REASONING_EFFORT':'xhigh','QK_PREFILL_CHUNK':'512',
               'QK_FLASH_PREFILL_LAST':'1' if mode in ('last','combined') else '0',
               'QK_FLASH_ATTN_BATCH':'vec4' if mode in ('vec4','combined') else 'baseline'}
        require(ctrl['actual_environment']==env and ctrl['default_promotion'] is False,'wrong runtime configuration')
        expected_command = ['/home/ryan/qk-last-head-IIfPmw/rust-release/release/server',
            '--model','/home/ryan/models/Qwen3.8-Flash-Next-Uncensored-Q5_K_M/Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf',
            '--engine-lib','/home/ryan/qk-last-head-IIfPmw/build/libqk.so',
            '--host','127.0.0.1','--port','8194','--slots','1','--ctx','32768',
            '--chunk','1','--queue','4','--local-driver','--chat-template','auto']
        require(ctrl['actual_command']==expected_command,'wrong actual command')
        require(ctrl['build_sha256']==sha(folder/'http-build.json') and ctrl['full_gate_sha256']==sha(folder/'gate.jsonl'),
                'wrong frozen evidence hash')
        seeded.add(ctrl['sampled_token_sha256'])
        for group,sizes,count in [('short',[128,512,2048,8192],128),('long',[16384],512)]:
            rows = read(str(prefix)+f'.{group}.jsonl')
            require(rows[0]['type']=='run_start' and rows[-1]['type']=='run_complete','HTTP matrix not complete')
            header = rows[0]
            require(header['sizes']==sizes and header['repetitions']==1 and header['context']==32768 and
                    header['requested_output_tokens']==count,'wrong HTTP workload')
            meta = header['metadata']
            require(header['config_sha256']==pm.digest(meta) and meta['build']==expected_build,'wrong bound metadata')
            require(meta['actual_environment']==env and meta['mode']==mode and meta['launch_index']==index and
                    meta['experiment']=='native-combined-http-v1' and meta['precision']=='native F32' and
                    meta['mtp'] is False and meta['default_promotion'] is False and meta['context']==32768 and
                    meta['slots']==1 and meta['prefill_chunk']==512,'wrong HTTP backend')
            require(meta['actual_command']==expected_command and meta['server_command']==ctrl['server_command']
                    and meta['order']==ORDER and meta['http_returncode']==0
                    and meta['sampled_token_sha256']==ctrl['sampled_token_sha256'],'metadata/controller mismatch')
            cells = [r for r in rows if r['type'] in ('prefill','decode')]
            require([(r['type'],r['prompt_tokens'],r['repetition']) for r in cells]==
                    [(kind,size,1) for size in sizes for kind in ('prefill','decode')],'missing/reordered cells')
            for row in cells:
                require(all(row[k]==header[k] for k in ('run_id','backend','model_id','fixture_sha256',
                            'context','config_sha256','requested_output_tokens')),'cell identity mismatch')
                require(row['server_prompt_tokens']==row['prompt_tokens'],'server token count differs')
                require(row['prompt_sha256']==pm.digest(pm.prompt_for(fixture,row['prompt_tokens'])),'wrong prompt')
                key = (row['type'],row['prompt_tokens'],count)
                if row['type']=='prefill':
                    value = row['output_tokens']
                    require(value==[16] and pm.positive_number(row['prefill_plus_one_token_seconds']),'bad probe')
                else:
                    require(row['exact_output_length'] and row['coherent_counting_prefix'] is True and
                            row['streamed_tokens']==count and row['complete_chunk_token_counts'],'bad counting stream')
                    for field in ('ttft_seconds','stream_total_seconds','decode_tokens_per_second'):
                        require(pm.positive_number(row[field]),'bad HTTP timing')
                    require(row['stream_total_seconds']>=row['ttft_seconds'],'inconsistent stream time')
                    value = [row['output_token_sha256'],row['output_text_sha256']]
                    require(all(re.fullmatch('[0-9a-f]{64}',s) for s in value),'invalid output digest')
                    measurements.append(dict(mode=mode,launch_index=index,**row))
                if key not in compared:
                    compared[key]=value
                require(compared[key]==value,'cross-configuration output differs')
        code_rows = read(str(prefix)+'.code.jsonl')
        require(code_rows[0]['type']=='metadata' and code_rows[-1]['type']=='complete','code probes incomplete')
        cm = code_rows[0]
        require(cm['experiment']=='native-code-stream-v1' and cm['max_output_tokens']==256
                and cm['generated_code_executed'] is False
                and re.fullmatch('[0-9a-f]{64}',cm['workload_sha256']),'wrong code workload')
        if code_metadata is None:
            code_metadata=cm
        require(cm==code_metadata,'code workload changed across launches')
        tests = [r for r in code_rows if r['type']=='code']
        require([r['name'] for r in tests]==['python-lru-review','vulkan-attention-review'],'missing code probe')
        for row in tests:
            require(row['complete_chunk_token_counts'] and 16<=row['streamed_tokens']<=256
                    and row['requested_output_tokens']==256
                    and row['server_prompt_tokens']==row['prompt_tokens'],'invalid code counts')
            require(all(pm.positive_number(row[k]) for k in ('ttft_seconds','decode_tokens_per_second',
                        'stream_total_seconds')) and row['stream_total_seconds']>=row['ttft_seconds'],'invalid code timings')
            key = row['name']
            value = [row['prompt_sha256'],row['output_token_sha256'],row['output_text_sha256'],row['streamed_tokens']]
            if key not in code:
                code[key]=value
            require(code[key]==value,'code output differs across launches')
            code_measurements.append(dict(mode=mode,launch_index=index,**row))
    require(len(seeded)==1 and all(re.fullmatch('[0-9a-f]{64}',x) for x in seeded),'seeded outputs differ')
    summary = []
    for mode in ORDER[:4]:
        for size in (128,512,2048,8192,16384):
            cells = [r for r in measurements if r['mode']==mode and r['prompt_tokens']==size]
            require(len(cells)==2,'missing independent launch repetition')
            summary.append(dict(mode=mode,prompt_tokens=size,output_tokens=512 if size==16384 else 128,
                **{field:stats([r[field] for r in cells]) for field in
                   ('ttft_seconds','decode_tokens_per_second','stream_total_seconds')}))
    code_summary = []
    for mode in ORDER[:4]:
        for name in ('python-lru-review','vulkan-attention-review'):
            cells = [r for r in code_measurements if r['mode']==mode and r['name']==name]
            require(len(cells)==2,'missing code launch repetition')
            code_summary.append(dict(mode=mode,name=name,prompt_tokens=cells[0]['prompt_tokens'],
                output_tokens=cells[0]['streamed_tokens'],
                **{field:stats([r[field] for r in cells]) for field in
                   ('ttft_seconds','decode_tokens_per_second','stream_total_seconds')}))
    return dict(result='PASS',order=ORDER,launches=8,paired_counting_cells=80,code_probes=16,
                sampled_token_sha256=next(iter(seeded)),summary=summary,resources=resources,
                code_summary=code_summary,
                caveat='Two independent launches per mode, symmetric order; synthetic counting and two unscored code reviews.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('gate','profile','http'))
    parser.add_argument('folder',type=Path)
    args = parser.parse_args()
    if args.mode=='gate':
        result = validate_combined(read(args.folder/'gate.jsonl'),read('/home/ryan/qk-last-head-full-VjRrMt/gate.jsonl'))
    elif args.mode=='profile':
        result = validate_profile(read(args.folder/'profile.jsonl'),(args.folder/'profile.log').read_text())
    else:
        result = validate_http(args.folder)
    if args.mode!='http':
        result['resources'] = validate_resources(json.loads((args.folder/f'{args.mode}.controller.json').read_text()),
                                                 read(args.folder/f'{args.mode}.observer.jsonl'),True)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
