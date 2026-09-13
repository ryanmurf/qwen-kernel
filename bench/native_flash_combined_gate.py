#!/usr/bin/env python3
"""Full-model vec4 + last-output equivalence against a frozen same-build gate.

Runs all/last/last/all, including every long-prefill boundary and serial tail.
The reference must be the completed baseline-attention gate from the exact
same native library and shader bytes. This is not an external quality oracle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import native_last_head_full as full
import audit_native_last_head_full as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'model', 'fixture', 'reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    reference = [json.loads(line) for line in args.reference.read_text().splitlines()]
    audit.validate_gate(reference)
    header = reference[0]
    full.require(header['library_sha256'] == full.partial.digest(args.library), 'different reference library')
    full.require(header['model'] == full.old.model_identity(str(args.model)), 'different reference model')
    current_shaders = full.old.shader_identity(os.environ['QK_SHADER_DIR'])
    full.require(header['shaders'] == current_shaders, 'different reference shaders')
    fixture = full.pm.load_fixture(args.fixture)
    full.require(header['fixture_sha256'] == fixture['sha256'], 'different reference fixture')
    expected = {r['prompt_rows']: r['logit_sha256'] for r in reference
                if r['type'] == 'case_pass' and r['iteration'] == 0}
    full.require(len(expected) == 15, 'incomplete reference')
    full.ENV['QK_FLASH_ATTN_BATCH'] = 'vec4'
    memory, env, cg = full.admission()
    sys.stdout.flush()
    full.OUT = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    identity = lambda: {'library': full.partial.digest(args.library),
                        'shaders': full.old.shader_identity(env['QK_SHADER_DIR']),
                        'model': full.old.model_identity(str(args.model)),
                        'reference': full.partial.digest(args.reference),
                        'harness': full.partial.digest(__file__)}
    frozen = identity()
    full.emit(type='metadata', experiment='native-combined-gate-v1', context=32768,
              chunk=512, environment=env, admission=memory, cgroup=cg, identity=frozen,
              reference_comparison='frozen same-library baseline attention; all four passes')
    runner = full.Runner(args.library, args.model)
    try:
        cases = [(n, False, fixture['source_ids'][:n+66]) for n in
                 (1, 2, 63, 64, 65, 127, 128, 129, 309, 511, 512, 513, 1024)]
        for size in (8192, 16384):
            cases.append((size+309, True, full.pm.prompt_for(fixture, size) +
                          [(1000 + 7*i) % 248320 for i in range(341)]))
        compared = 0
        for n, chunked, tokens in cases:
            plan = full.row_plan(n, chunked)
            full.require(len(plan) == len(expected[n]), 'reference row coverage mismatch')
            for iteration, last in enumerate((False, True, True, False)):
                hashes = []
                for index, (base, width) in enumerate(plan):
                    payload, greedy, elapsed = runner.step(tokens[base:base+width], base, last)
                    digest = hashlib.sha256(payload).hexdigest()
                    full.require(digest == expected[n][index],
                                 f'non-exact combined logits n={n}, iteration={iteration}, base={base}')
                    hashes.append(digest)
                    compared += 1
                    full.emit(type='row', prompt_rows=n, iteration=iteration, last_only=last,
                              row=index, base=base, width=width, greedy=greedy, seconds=elapsed,
                              logit_sha256=digest)
                full.emit(type='case_pass', prompt_rows=n, chunked=chunked,
                          iteration=iteration, last_only=last, logit_sha256=hashes)
        full.require(identity() == frozen, 'identity changed')
        full.emit(type='result', result='PASS', cases=len(cases), compared_rows=compared,
                  bit_exact=True, identities_unchanged=True, full_model=True)
    finally:
        runner.close()
        full.emit(type='closed', mem_available=full.partial.mem_available(), cgroup=full.partial.cgroup())


if __name__ == '__main__':
    main()
