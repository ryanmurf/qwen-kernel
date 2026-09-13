#!/usr/bin/env python3
"""Instrumented full-model 2K/8K/16K prefill and decode, never API throughput.

Uses the installed last-output ABI, F32 baseline attention and 512-token chunks.
Native per-dispatch fences perturb execution; preserve wall and GPU times
separately. Marker-delimited stderr retains all native attribution evidence.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import native_last_head_full as full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'model', 'fixture'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    full.ENV.update(QK_FLASH_PROFILE='1', QK_FLASH_REPLAY='0', QK_FLASH_TIMING='1')
    memory, env, cg = full.admission()
    sys.stdout.flush()
    full.OUT = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    fixture = full.pm.load_fixture(args.fixture)
    full.emit(type='metadata', experiment='native-long-profile-v1', context=32768, chunk=512,
              environment=env, admission=memory, cgroup=cg, library_sha256=full.partial.digest(args.library),
              shader_identity=full.old.shader_identity(env['QK_SHADER_DIR']), fixture_sha256=fixture['sha256'],
              caveat='Every dispatch fenced; GPU attribution is instrumented, not serving throughput.')
    runner = full.Runner(args.library, args.model)

    def step(tokens, base, phase, size, index):
        full.require(full.partial.mem_available() >= 8 * full.partial.GIB, 'memory floor')
        n = len(tokens)
        inp = (full.U * n)(*tokens)
        out = (full.U * 3)(0xD15EA5ED, 0xD15EA5ED, 0xD15EA5ED)
        marker = dict(phase=phase, prompt_tokens=size, index=index, base=base, width=n)
        print('[campaign begin] ' + json.dumps(marker), file=sys.stderr, flush=True)
        start = time.perf_counter()
        rc = runner.lib.qk_stage_run_last(runner.engine, 0, inp, None, n, base,
                                        C.cast(C.byref(out, 4), C.POINTER(full.U)))
        elapsed = time.perf_counter() - start
        print('[campaign end] ' + json.dumps(marker), file=sys.stderr, flush=True)
        full.require(rc == 0 and out[0] == out[2] == 0xD15EA5ED and out[1] < 248320,
                     'profile step failed or canary changed')
        full.emit(type='step', **marker, seconds=elapsed, greedy=int(out[1]))
        return int(out[1])

    try:
        for size in (2048, 8192, 16384):
            tokens = full.pm.prompt_for(fixture, size)
            for index, base in enumerate(range(0, size, 512)):
                token = step(tokens[base:base+512], base, 'prefill', size, index)
            generated = []
            for index in range(16):
                generated.append(token)
                token = step([token], size+index, 'decode', size, index)
            full.emit(type='case_pass', prompt_tokens=size, decode_steps=16,
                      generated_sha256=full.pm.digest(generated), generated=generated)
        full.emit(type='result', result='PASS', cases=3)
    finally:
        runner.close()
        full.emit(type='closed', mem_available=full.partial.mem_available(), cgroup=full.partial.cgroup())


if __name__ == '__main__':
    main()
