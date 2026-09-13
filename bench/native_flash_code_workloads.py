#!/usr/bin/env python3
"""Two deterministic code-review streams; no generated code is executed.

These are timing/output-equivalence probes, not a scored coding benchmark.
Natural early EOS is retained with its actual token count, never dropped.
"""
import argparse
import json
from pathlib import Path
import time

import prefill_matrix as pm

ROOT = Path(__file__).resolve().parents[1]


def prompts():
    cache = '''from collections import OrderedDict

class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, key):
        return self.items.get(key)

    def put(self, key, value):
        if len(self.items) >= self.capacity:
            self.items.popitem(last=False)
        self.items[key] = value
'''
    return [
        ('python-lru-review', 'Review this intended LRU cache. Identify concrete correctness bugs, '
         'give short counterexamples, and describe a corrected implementation and tests.\n\n' + cache),
        ('vulkan-attention-review', 'Review this Vulkan compute shader for performance at 16384 keys '
         'with hQ=24, hKV=2 and dh=256. Explain parallelism, memory accesses and synchronization, '
         'then propose specific optimizations and distinguish exact arithmetic changes from '
         'changes needing numerical validation.\n\n' + (ROOT/'shaders/fa_attn_srv.comp').read_text())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x', buffering=1) as stream:
        def emit(**row):
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        emit(type='metadata', experiment='native-code-stream-v1', max_output_tokens=256,
             workload_sha256=pm.digest(prompts()), generated_code_executed=False,
             caveat='Unscored code-review probes; early EOS retained.')
        for name, user in prompts():
            prompt = ('<|im_start|>system\nYou are a careful code reviewer. Be specific.'
                      '<|im_end|>\n<|im_start|>user\n'+user+
                      '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            tokens = pm.tokenize(args.url,prompt,True,60)
            result = pm.measure_stream(args.url,tokens,256,600)
            if not (result['complete_chunk_token_counts'] and result['streamed_tokens'] >= 16
                    and pm.positive_number(result['decode_tokens_per_second'])):
                raise ValueError('invalid code stream: '+name)
            emit(type='code', name=name, prompt_tokens=len(tokens), prompt_sha256=pm.digest(tokens), **result)
        emit(type='complete')


if __name__ == '__main__':
    main()
