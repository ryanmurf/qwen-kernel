#!/usr/bin/env python3
"""Explicit fresh/repeat/follow-up llama.cpp API matrix; no model management.

Fresh means no reused KV, NOT cold model/OS page cache. Exact repeats are
best-case cached requests, not a representative agent workload. The appended
turn includes the preceding generated IDs. No logprobs or synthetic acceptance.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import uuid

import prefill_matrix as pm

FOLLOWUP = ('<|im_end|>\n<|im_start|>user\nStart again. Write the integers from 1 to 1000, '
            'separated by commas, without explanation.<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')
KINDS = ('fresh', 'repeat', 'followup')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def body(prompt, count, reuse):
    return {'prompt': prompt, 'n_predict': count, 'stream': True,
            'temperature': 0, 'seed': 0, 'cache_prompt': reuse,
            'return_tokens': True, 'id_slot': 0}


def measure(url, prompt, count, reuse, timeout):
    started = utc(); start = time.monotonic()
    first = last = None
    first_count = 0
    texts, ids = [], []
    final = None
    events = 0
    with pm.post(url, '/completion', body(prompt, count, reuse), timeout) as response:
        for event in pm.sse(response):
            require(isinstance(event, dict) and not event.get('error'), 'stream error')
            if event.get('stop'):
                require(final is None, 'duplicate final record')
                final = event
                continue
            tokens = event.get('tokens')
            content = event.get('content', '')
            if not content and not tokens:
                continue
            require(final is None, 'tokens after final record')
            pm.token_ids(tokens)
            require(isinstance(content, str), 'invalid stream text')
            now = time.monotonic()
            if first is None:
                first = now; first_count = len(tokens)
            last = now; events += 1
            ids.extend(tokens); texts.append(content)
    require(first is not None and final is not None, 'incomplete stream')
    output = ''.join(texts)
    compact = re.sub(r'\s+', '', output)
    coherent = bool(compact) and ','.join(str(i) for i in range(1, 1001)).startswith(compact)
    elapsed = last - first
    rate = (len(ids) - first_count) / elapsed if elapsed > 0 and len(ids) > first_count else None
    timings = final.get('timings') or {}
    return {'started_utc': started, 'ended_utc': utc(), 'ttft_seconds': first - start,
            'stream_total_seconds': time.monotonic() - start, 'decode_tokens_per_second': rate,
            'streamed_tokens': len(ids), 'first_event_tokens': first_count, 'events': events,
            'exact_output_length': len(ids) == count, 'coherent_counting_prefix': coherent,
            'output_tokens': ids, 'output_preview': output[:256],
            'output_token_sha256': pm.digest(ids),
            'output_text_sha256': hashlib.sha256(output.encode()).hexdigest(),
            'server_timings': timings, 'server_retained_tokens': final.get('tokens_cached'),
            'server_evaluated_prompt_tokens': final.get('tokens_evaluated'),
            'server_slot': final.get('id_slot'), 'stop_type': final.get('stop_type'),
            'generation_settings': final.get('generation_settings')}


def audit_cell(row, count):
    require(row['cache_condition'] in KINDS, 'unknown cache condition')
    require(row['exact_output_length'] is True and row['streamed_tokens'] == count,
            'short/uncounted output')
    pm.token_ids(row['output_tokens'])
    require(len(row['output_tokens']) == count and pm.digest(row['output_tokens']) == row['output_token_sha256'],
            'output token hash mismatch')
    require(row['coherent_counting_prefix'] is True, 'incoherent counting output')
    require(all(pm.positive_number(row[k]) for k in ('ttft_seconds', 'stream_total_seconds',
                                                   'decode_tokens_per_second')), 'invalid timing')
    timings = row['server_timings']
    cached, processed = timings.get('cache_n'), timings.get('prompt_n')
    require(type(cached) is int and type(processed) is int and cached >= 0 and processed >= 0,
            'missing actual cache/evaluation counts')
    require(cached + processed == row['prompt_tokens'], 'cache + processed != input length')
    require(timings.get('predicted_n') == count, 'server output count differs')
    require(row['server_slot'] == 0, 'wrong slot')
    if row['cache_condition'] == 'fresh':
        require(cached == 0, 'fresh request reused KV')
    # A cache miss on a reuse request is retained and labelled, never silently
    # promoted to a cached speed result; recurrent checkpoints may limit reuse.
    return cached > 0


def run(args):
    require(args.confirm_exclusive, 'confirm an exclusive Halo-only server')
    require(1 <= args.repetitions <= 10 and 2 <= args.decode_tokens <= 1024, 'invalid workload')
    fixture = pm.load_fixture(args.fixture)
    sizes = args.sizes or fixture['sizes']
    require(sizes == sorted(set(sizes)) and all(n in fixture['sizes'] for n in sizes), 'invalid sizes')
    require(pm.tokenize(args.url, fixture['source_text'], True, args.timeout) == fixture['source_ids'],
            'source tokenizer differs')
    require(pm.tokenize(args.url, fixture['suffix_text'], False, args.timeout) == fixture['suffix_ids'],
            'suffix tokenizer differs')
    followup = pm.tokenize(args.url, FOLLOWUP, False, args.timeout)
    require(max(sizes) + 2*args.decode_tokens + len(followup) < args.context, 'insufficient context')
    metadata = json.loads(Path(args.metadata).read_text())
    base = {'run_id': str(uuid.uuid4()), 'backend': args.backend,
            'fixture_sha256': fixture['sha256'], 'model_id': fixture['model_id'],
            'config_sha256': pm.digest(metadata), 'context': args.context,
            'requested_output_tokens': args.decode_tokens}
    with Path(args.output).open('x', buffering=1) as out:
        def emit(kind, **fields):
            row = {'type': kind, **base, **fields}
            out.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row, allow_nan=False), flush=True)
            return row
        emit('run_start', utc=utc(), sizes=sizes, repetitions=args.repetitions,
             metadata=metadata, followup_ids=followup, followup_sha256=pm.digest(followup),
             cache_procedure='fresh: cache_prompt=false; repeat and appended turn: true; slot 0; actual cache_n recorded; no OS cache flush')
        try:
            for size in sizes:
                prompt = pm.prompt_for(fixture, size)
                for rep in range(1, args.repetitions+1):
                    previous = None
                    for condition in KINDS:
                        tokens = prompt if condition != 'followup' else prompt + previous['output_tokens'] + followup
                        result = measure(args.url, tokens, args.decode_tokens, condition != 'fresh', args.timeout)
                        row = emit('measurement', base_prompt_tokens=size, prompt_tokens=len(tokens),
                                   prompt_sha256=pm.digest(tokens), repetition=rep, cache_condition=condition, **result)
                        hit = audit_cell(row, args.decode_tokens)
                        previous = result
                        print(f"cell {size} {rep} {condition}: cache_hit={hit}", file=__import__('sys').stderr, flush=True)
            emit('run_complete', utc=utc())
        except BaseException as error:
            emit('run_error', utc=utc(), error=f'{type(error).__name__}: {error}')
            raise


def audit(rows):
    require(len(rows) >= 5 and rows[0]['type'] == 'run_start' and rows[-1]['type'] == 'run_complete', 'incomplete matrix')
    head = rows[0]
    require(pm.digest(head['metadata']) == head['config_sha256'], 'metadata hash mismatch')
    require(pm.digest(head['followup_ids']) == head['followup_sha256'], 'followup hash mismatch')
    wanted = [(n, r, k) for n in head['sizes'] for r in range(1, head['repetitions']+1) for k in KINDS]
    actual = [(r.get('base_prompt_tokens'), r.get('repetition'), r.get('cache_condition')) for r in rows[1:-1]]
    require(actual == wanted, 'missing/reordered/extra cells')
    for row in rows[1:]:
        require(all(row[k] == head[k] for k in ('run_id', 'backend', 'fixture_sha256', 'model_id',
                    'config_sha256', 'context', 'requested_output_tokens')), 'mixed run identity')
    for row in rows[1:-1]:
        require(row['type'] == 'measurement', 'unexpected row type')
        audit_cell(row, head['requested_output_tokens'])
    for i in range(1, len(rows)-1, 3):
        fresh, repeat, followup = rows[i:i+3]
        require(fresh['prompt_sha256'] == repeat['prompt_sha256'] and fresh['prompt_tokens'] == repeat['prompt_tokens'],
                'repeat prompt differs')
        require(followup['prompt_tokens'] == repeat['prompt_tokens'] + len(repeat['output_tokens']) + len(head['followup_ids']),
                'incorrect followup length')
    return head, rows[1:-1]


def summarize(rows):
    head, cells = audit(rows)
    groups = []
    for size in head['sizes']:
        for condition in KINDS:
            values = [r for r in cells if r['base_prompt_tokens'] == size and r['cache_condition'] == condition]
            groups.append({'base_prompt_tokens': size, 'cache_condition': condition, 'samples': len(values),
                           'actual_cache_hits': sum(r['server_timings']['cache_n'] > 0 for r in values),
                           **{key: pm.stats([r[key] for r in values]) for key in ('ttft_seconds', 'decode_tokens_per_second')},
                           'unique_output_hashes': len({r['output_token_sha256'] for r in values})})
    return {'matrix_valid': True, 'backend': head['backend'], 'rows': groups,
            'scope': 'counting workload; not broad quality; fresh KV is not cold OS/file cache'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    for name in ('url', 'backend', 'fixture', 'metadata', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--context', type=int, default=65536)
    p.add_argument('--sizes', type=int, nargs='+')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--decode-tokens', type=int, default=128)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--confirm-exclusive', action='store_true')
    sub.add_parser('summarize').add_argument('results')
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    else:
        rows = [json.loads(s) for s in Path(args.results).read_text().splitlines() if s.strip()]
        print(json.dumps(summarize(rows), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
