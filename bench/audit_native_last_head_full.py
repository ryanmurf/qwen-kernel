#!/usr/bin/env python3
"""Strict full-model last-head gate and same-token HTTP A/B audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from native_last_head_full import ENV, row_plan

SHORT = [1, 2, 63, 64, 65, 127, 128, 129, 309, 511, 512, 513, 1024]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def validate_gate(records):
    require(records[0]['type'] == 'metadata' and records[-1]['type'] == 'closed', 'gate not closed')
    meta = records[0]
    require(meta['experiment'] == 'native-last-head-full-v1' and meta['context'] == 32768
            and meta['chunk'] == 512 and meta['admission']['admitted'], 'wrong gate configuration')
    env = meta['environment']
    require(env == {**ENV, 'QK_SHADER_DIR': env.get('QK_SHADER_DIR')}
            and Path(env['QK_SHADER_DIR']).is_absolute(), 'wrong full-model gate environment')
    expected = []
    hashes = {}
    greedy = {}
    for size in SHORT + [8501, 16693]:
        plan = row_plan(size, size not in SHORT)
        for iteration, last in enumerate((False, True, True, False)):
            for index, (base, width) in enumerate(plan):
                expected.append(('row', size, iteration, last, index, base, width))
            expected.append(('case_pass', size, iteration, last))
    observed = []
    for rec in records:
        if rec['type'] == 'row':
            observed.append(tuple(rec[k] for k in ('type','prompt_rows','iteration','last_only','row','base','width')))
            require(0 <= rec['greedy'] < 248320 and rec['seconds'] > 0, 'invalid row')
            key = (rec['prompt_rows'], rec['row'])
            if key not in greedy:
                greedy[key] = rec['greedy']
            require(rec['greedy'] == greedy[key], 'cross-policy/reset greedy differs')
        elif rec['type'] == 'case_pass':
            observed.append(tuple(rec[k] for k in ('type','prompt_rows','iteration','last_only')))
            size = rec['prompt_rows']
            require(rec['compared_rows'] == len(row_plan(size, size not in SHORT)), 'missing logit rows')
            require(len(rec['logit_sha256']) == rec['compared_rows'], 'missing logit hashes')
            require(all(re.fullmatch('[0-9a-f]{64}', h) for h in rec['logit_sha256']), 'invalid logit hash')
            if size not in hashes:
                hashes[size] = rec['logit_sha256']
            require(rec['logit_sha256'] == hashes[size], 'non-exact cross-policy/reset logits')
    require(observed == expected, 'incomplete or reordered gate')
    result = [rec for rec in records if rec['type'] == 'result']
    require(len(result) == 1 and result[0]['result'] == 'PASS' and result[0]['cases'] == 15
            and result[0]['compared_rows'] == 498 and result[0]['bit_exact']
            and result[0]['identities_unchanged'], 'missing successful full gate')
    return {'result':'PASS','full_model':True,'cases':15,'compared_logit_rows':498,
            'unique_baseline_rows':166,'library_sha256':meta['library_sha256'],
            'fixture_sha256':meta['fixture_sha256'],'http_validation':False}


def validate_http(all_rows, last_rows):
    sizes = [128, 512, 2048, 8192, 16384]
    paired = []
    for mode, rows in (('all', all_rows), ('last', last_rows)):
        require(rows[0]['type'] == 'run_start' and rows[-1]['type'] == 'run_complete', 'HTTP run incomplete')
        first = rows[0]
        require(first['sizes'] == sizes and first['repetitions'] == 3 and first['context'] == 32768
                and first['requested_output_tokens'] == 128, 'wrong HTTP matrix')
        metadata = first['metadata']
        require(metadata['mode'] == mode and metadata['http_returncode'] == 0, 'HTTP compatibility failed')
        require(metadata['actual_environment']['QK_FLASH_PREFILL_LAST'] == ('1' if mode == 'last' else '0'), 'wrong last-output flag')
        expected = [(kind, size, repetition) for size in sizes for repetition in (1,2,3) for kind in ('prefill','decode')]
        require([(r['type'],r['prompt_tokens'],r['repetition']) for r in rows[1:-1]] == expected, 'HTTP cells incomplete/reordered')
        for rec in rows:
            for key in ('run_id','backend','model_id','fixture_sha256','context','config_sha256','requested_output_tokens'):
                require(rec[key] == first[key], 'HTTP identity changed within run')
        for rec in rows[1:-1]:
            require(rec['server_prompt_tokens'] == rec['prompt_tokens'], 'not a fresh full prefill')
            if rec['type'] == 'decode':
                require(rec['exact_output_length'] and rec['complete_chunk_token_counts'] and
                        rec['coherent_counting_prefix'] and rec['streamed_tokens'] == 128 and
                        rec['ttft_seconds'] > 0 and rec['decode_tokens_per_second'] > 0, 'invalid decode cell')
            else:
                require(rec['output_tokens'] == [16] and rec['prefill_plus_one_token_seconds'] > 0, 'invalid prefill probe')
    a, b = all_rows[0], last_rows[0]
    for key in ('fixture_sha256','model_id','context','requested_output_tokens'):
        require(a[key] == b[key], 'pair workload differs')
    for key in ('build','full_gate_sha256','sampled_token_sha256','precision','mtp','slots','prefill_chunk'):
        require(a['metadata'][key] == b['metadata'][key], f'pair metadata differs: {key}')
    ea, eb = (dict(r['metadata']['actual_environment']) for r in (a,b))
    ea.pop('QK_FLASH_PREFILL_LAST'); eb.pop('QK_FLASH_PREFILL_LAST')
    require(ea == eb, 'environment differs beyond the selected flag')
    for x, y in zip(all_rows[1:-1], last_rows[1:-1]):
        require(x['prompt_sha256'] == y['prompt_sha256'], 'paired prompt differs')
        if x['type'] == 'decode':
            require(x['output_token_sha256'] == y['output_token_sha256'] and
                    x['output_text_sha256'] == y['output_text_sha256'], 'paired output differs')
    for size in sizes:
        item = {'prompt_tokens':size}
        for kind, metric in [('prefill','prefill_plus_one_token_seconds'),('decode','ttft_seconds'),('decode','decode_tokens_per_second')]:
            for label, rows in [('all',all_rows),('last',last_rows)]:
                values = [r[metric] for r in rows if r['type'] == kind and r['prompt_tokens'] == size]
                item[label+'_'+metric] = {'median':statistics.median(values),'min':min(values),'max':max(values)}
        item['ttft_speedup'] = item['all_ttft_seconds']['median']/item['last_ttft_seconds']['median']
        paired.append(item)
    return {'result':'PASS','paired_prompt_cells':30,'paired_decode_outputs':15,
            'sampled_outputs_exact':True,'precision':'native F32','results':paired,
            'caveat':'fixed-order launches, three repetitions; synthetic counting, not general quality'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate', type=Path)
    parser.add_argument('--all-http', type=Path)
    parser.add_argument('--last-http', type=Path)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines()]
    result = validate_gate(read(args.gate))
    result['gate_sha256'] = hashlib.sha256(args.gate.read_bytes()).hexdigest()
    require(bool(args.all_http) == bool(args.last_http), 'need both HTTP runs')
    if args.all_http:
        a, b = read(args.all_http), read(args.last_http)
        require(a[0]['metadata']['full_gate_sha256'] == result['gate_sha256'], 'HTTP gate binding differs')
        result['http'] = validate_http(a,b)
        result['http_validation'] = True
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
