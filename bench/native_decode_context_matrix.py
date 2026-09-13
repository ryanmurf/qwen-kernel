#!/usr/bin/env python3
"""Fresh-KV decode curve; single prefill per measured stream, no probe/cache hit.

No server lifecycle actions. One verified, exclusively owned server required.
31K leaves room for the 512-token answer in a 32K allocation. Counting is a
synthetic timing workload, not a quality evaluation. Use separate full-logit
and API checks and counterbalanced independent model launches.
"""
import argparse
import json
from pathlib import Path

import prefill_matrix as pm

SIZES = (128, 512, 2048, 8192, 16384, 31744)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('serial', 'loads'), required=True)
    parser.add_argument('--launch', type=int, required=True)
    args = parser.parse_args()
    fixture = pm.load_fixture(args.fixture)
    prompts = {n:pm.prompt_for(fixture,n) for n in SIZES}
    with args.output.open('x', buffering=1) as out:
        def emit(**row):
            print(json.dumps(row,allow_nan=False),file=out,flush=True)
        emit(type='metadata', mode=args.mode, launch=args.launch, context=32768,
             sizes=SIZES, fixture_sha256=fixture['sha256'],
             protocol='one fresh-KV stream per size; no preliminary one-token probe; 128 output below16K,512 at16K/31K',
             prompt_sha256={n:pm.digest(p) for n,p in prompts.items()},
             harness_sha256=pm.hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        for n,prompt in prompts.items():
            count=512 if n>=16384 else 128
            result=pm.measure_stream(args.url,prompt,count,1200)
            if not result['exact_output_length'] or not result['coherent_counting_prefix']:
                emit(type='failed', prompt_tokens=n, **result)
                raise RuntimeError('invalid counting completion')
            emit(type='decode',prompt_tokens=n,**result)
        emit(type='result', result='PASS', cases=len(SIZES))


if __name__=='__main__': main()
