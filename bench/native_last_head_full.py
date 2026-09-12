#!/usr/bin/env python3
"""Full 48-layer native last-output gate, one exclusive bounded GPU load.

Same library and precision, both ABIs, real benchmark token prefixes plus
explicit teacher-forced tails. Checks exact full logits, top-k, reset and
continuation. Not an external oracle or HTTP throughput benchmark.
"""
import argparse
import ctypes as C
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import native_last_head as partial
import prefill_matrix as pm

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('model_headroom', ROOT / 'deploy/check-model-headroom.py')
headroom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(headroom)
spec = importlib.util.spec_from_file_location('qwen4_reference_helpers', ROOT / 'tests/gpu_qwen4_attn_split.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
U, F, P = C.c_uint32, C.c_float, C.c_void_p
require = partial.require
OUT = sys.stdout
ENV = {'QK_NATIVE_FLASH': '1', 'QK_DEVICE_PCI': '0000:c1:00.0',
       'QK_FLASH_BATCH': '512', 'QK_FLASH_COOPMAT': '0', 'QK_FLASH_GEMM': 'baseline',
       'QK_FLASH_ATTN_BATCH': 'baseline', 'QK_ATTN_DECODE': 'serial', 'QK_PLE_PREFETCH': '0'}


def emit(**value):
    print(json.dumps(value, allow_nan=False), file=OUT, flush=True)


def row_plan(n, chunked=False):
    rows = [(i, min(512, n-i)) for i in range(0, n, 512)] if chunked else [(0, n)]
    for width in ([1] * 32 if chunked else [1, 2, 63]):
        rows.append((n, width))
        n += width
    return rows


def admission():
    env = {k: v for k, v in os.environ.items() if k.startswith('QK_')}
    require(env == {**ENV, 'QK_SHADER_DIR': env.get('QK_SHADER_DIR')}, 'unexpected/missing QK override')
    result = headroom.assess(headroom.parse_meminfo(Path('/proc/meminfo').read_text()))
    require(result['admitted'], str(result))
    cg = partial.cgroup()
    require(int(cg['memory.high']) <= 24 * partial.GIB and int(cg['memory.max']) <= 32 * partial.GIB
            and int(cg['memory.swap.max']) <= partial.GIB // 2, 'bounded dedicated cgroup required')
    for pci in ('0000:c1:00.0', '0000:68:00.0'):
        path = Path('/sys/bus/pci/devices') / pci
        require(int((path / 'mem_info_gtt_used').read_text()) < partial.GIB, 'GPU GTT not drained')
    require(Path('/sys/bus/pci/devices/0000:c1:00.0/device').read_text().strip() == '0x1586', 'not Halo')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            name = (proc / 'comm').read_text().strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        require(name not in ('qk', 'server', 'llama-server', 'llama-bench'), 'another inference process active')
    return result, env, cg


class Runner:
    def __init__(self, library, model):
        self.lib = old.load_lib(library)
        self.lib.qk_stage_run_last.argtypes = [P, U, C.POINTER(U), C.POINTER(F), U, U, C.POINTER(U)]
        self.lib.qk_stage_run_last.restype = C.c_int
        self.lib.qk_stage_topk.argtypes = [P, U, C.POINTER(U), C.POINTER(F)]
        self.lib.qk_stage_topk.restype = C.c_int
        err = C.create_string_buffer(4096)
        started = time.monotonic()
        self.engine = self.lib.qk_open(os.fsencode(model), C.byref(old.Config(1, 32768, 1)), err, len(err))
        require(self.engine, err.value.decode(errors='replace'))
        for name, expected in [('qk_layer_first', 0), ('qk_layer_end', 48), ('qk_n_embd', 10240), ('qk_n_vocab', 248320)]:
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = [P], U
            require(fn(self.engine) == expected, 'unexpected full-model architecture')
        emit(type='loaded', seconds=time.monotonic()-started, mem_available=partial.mem_available(), cgroup=partial.cgroup())

    def close(self):
        if self.engine:
            self.lib.qk_close(self.engine)
            self.engine = None

    def step(self, tokens, base, last):
        require(partial.mem_available() >= 8 * partial.GIB, 'live host headroom below 8 GiB')
        n = len(tokens)
        inp = (U * n)(*tokens)
        count = 1 if last else n
        out = (U * (count + 2))(*([0xD15EA5ED] * (count + 2)))
        dest = C.cast(C.byref(out, 4), C.POINTER(U))
        started = time.perf_counter()
        if last:
            rc = self.lib.qk_stage_run_last(self.engine, 0, inp, None, n, base, dest)
        else:
            rc = self.lib.qk_stage_run(self.engine, 0, inp, None, n, base, None, dest)
        elapsed = time.perf_counter() - started
        require(rc == 0, f'stage rc={rc}, base={base}, width={n}, last={last}')
        require(out[0] == out[count+1] == 0xD15EA5ED, 'output canary changed')
        require(all(v < 248320 for v in out[1:count+1]), 'invalid output ID')
        logits = (F * 248320)()
        require(self.lib.qk_stage_logits(self.engine, logits, 248320) == 0, 'full logits unavailable')
        require(all(math.isfinite(x) for x in logits), 'nonfinite full logits')
        expected = sorted(range(248320), key=lambda i: (-logits[i], i))[:20]
        ids, vals = (U * 20)(), (F * 20)()
        require(self.lib.qk_stage_topk(self.engine, 20, ids, vals) == 0, 'top-k unavailable')
        require(list(ids) == expected and ids[0] == out[count], 'top-k/greedy mismatch')
        require(all(vals[i] == logits[t] for i, t in enumerate(ids)), 'top-k values mismatch')
        return bytes(logits), int(out[count]), elapsed


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.flush()
    OUT = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    memory, env, cg = admission()
    fixture = pm.load_fixture(args.fixture)
    identities = lambda: dict(library_sha256=partial.digest(args.library), model=old.model_identity(str(args.model)),
                             shaders=old.shader_identity(env['QK_SHADER_DIR']), fixture_sha256=fixture['sha256'],
                             helpers={str(p.relative_to(ROOT)): partial.digest(p) for p in
                                      [Path(__file__), ROOT/'bench/native_last_head.py', ROOT/'bench/prefill_matrix.py',
                                       ROOT/'tests/gpu_qwen4_attn_split.py', ROOT/'deploy/check-model-headroom.py']})
    identity = identities()
    emit(type='metadata', experiment='native-last-head-full-v1', context=32768, chunk=512,
         admission=memory, environment=env, cgroup=cg, **identity)
    runner = Runner(args.library, args.model)
    try:
        source = fixture['source_ids']
        cases = [(n, False, source[:n+66]) for n in (1, 2, 63, 64, 65, 127, 128, 129, 309, 511, 512, 513, 1024)]
        for size in (8192, 16384):
            tokens = pm.prompt_for(fixture, size) + [(1000 + 7*i) % 248320 for i in range(309 + 32)]
            cases.append((size+309, True, tokens))
        checked_rows = 0
        for n, chunked, tokens in cases:
            plan = row_plan(n, chunked)
            expected = []
            for iteration, last in enumerate((False, True, True, False)):
                total = 0.0
                hashes = []
                for index, (base, width) in enumerate(plan):
                    payload, greedy, elapsed = runner.step(tokens[base:base+width], base, last)
                    total += elapsed
                    if iteration == 0:
                        expected.append((payload, greedy))
                    else:
                        require(expected[index] == (payload, greedy), f'non-exact logits n={n}, iteration={iteration}, base={base}')
                        checked_rows += 1
                    hashes.append(partial.hashlib.sha256(payload).hexdigest())
                    emit(type='row', prompt_rows=n, iteration=iteration, last_only=last,
                         row=index, base=base, width=width, greedy=greedy, seconds=elapsed)
                emit(type='case_pass', prompt_rows=n, chunked=chunked, iteration=iteration,
                     last_only=last, logit_sha256=hashes, stage_seconds=total, compared_rows=len(plan))
        require(identity == identities(), 'build/model/helper identities changed during run')
        emit(type='result', result='PASS', full_model=True, cases=len(cases),
             compared_rows=checked_rows, bit_exact=True, identities_unchanged=True,
             cgroup=partial.cgroup(), mem_available=partial.mem_available(), http_validation=False)
    finally:
        runner.close()
        emit(type='closed', mem_available=partial.mem_available(), cgroup=partial.cgroup())


if __name__ == '__main__':
    main()
