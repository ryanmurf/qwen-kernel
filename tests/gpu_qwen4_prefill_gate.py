#!/usr/bin/env python3
"""Bit-exact native prefill policy gate, including prefill-boundary logits.

Run dump twice in separate, guarded, exclusive Halo windows, then compare.
Unlike the decode-only replay gate, every prefill chunk's LAST position is
saved before decode can overwrite it. This is not every intermediate token's
logits, an external model oracle, or an API throughput benchmark. Every saved
row and its ABI greedy ID must also repeat exactly after a complete reset.
Historical gpu_qwen4_attn_split.py and its manifests remain unchanged.
"""
import argparse
from array import array
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('decode_gate', HERE / 'gpu_qwen4_attn_split.py')
old = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(old)
VERSION = 1
VOCAB = old.VOCAB
CHUNK = 512
BASE_ENV = {'QK_NATIVE_FLASH': '1', 'QK_DEVICE_NAME': 'STRIX_HALO',
            'QK_FLASH_COOPMAT': '0', 'QK_PLE_PREFETCH': '0',
            'QK_ATTN_DECODE': 'serial', 'QK_ATTN_CHUNK': '256'}
POLICIES = {'baseline': ('baseline', 'baseline'), 'vec4': ('baseline', 'vec4'),
            'compact': ('compact', 'baseline')}
REQUIRED = {'manifest_version', 'experiment', 'policy', 'out', 'dump_sha256',
            'ids', 'ids_sha256', 'n', 'tail', 'ctx', 'vocab', 'chunk', 'row_plan',
            'model', 'library_sha256', 'shaders', 'helper_sha256', 'knobs',
            'reset_exact', 'prefill_repeat_exact', 'tail_repeat_exact',
            'repeat_exact', 'all_finite', 'abi_greedy_exact', 'greedy_ids',
            'seconds', 'identities_unchanged'}
PER_RUN = {'policy', 'out', 'dump_sha256', 'knobs', 'greedy_ids', 'seconds'}

def require(ok, reason):
    if not ok: raise ValueError(reason)

def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

def plan(n, tail):
    require(type(n) is int and type(tail) is int and 1 <= tail <= min(512, n - 1), 'invalid n/tail')
    prefix = n - tail
    rows = [{'phase': 'clean', 'base': 0, 'width': 1}]
    rows += [{'phase': 'prefill', 'base': base, 'width': min(CHUNK, prefix - base)}
             for base in range(0, prefix, CHUNK)]
    rows += [{'phase': 'tail', 'base': base, 'width': 1} for base in range(prefix, n)]
    return rows

def row_properties(row, greedy, vocab):
    require(len(row) == vocab, 'wrong logit row width')
    finite = all(math.isfinite(v) for v in row)
    exact = type(greedy) is int and 0 <= greedy < vocab
    exact = finite and exact and greedy == max(range(vocab), key=row.__getitem__)
    return finite, exact

def capture(runner, ids, tail, vocab=VOCAB, progress=None):
    """Return first-pass row bytes plus metadata; compare every repeated row."""
    row_plan = plan(len(ids), tail)
    saved, greedy_ids = [], []
    checks = {'reset_exact': True, 'prefill_repeat_exact': True,
              'tail_repeat_exact': True, 'all_finite': True, 'abi_greedy_exact': True}
    for repeat in (False, True):
        for index, event in enumerate(row_plan):
            base, width = event['base'], event['width']
            greedy = runner.step(ids[base:base + width], base)
            row = runner.row()
            finite, exact = row_properties(row, greedy, vocab)
            checks['all_finite'] &= finite; checks['abi_greedy_exact'] &= exact
            payload = row.tobytes()
            require(len(payload) == vocab * 4, 'logits must be float32')
            if not repeat:
                saved.append(payload); greedy_ids.append(greedy)
            else:
                key = {'clean': 'reset_exact', 'prefill': 'prefill_repeat_exact', 'tail': 'tail_repeat_exact'}[event['phase']]
                checks[key] &= payload == saved[index] and greedy == greedy_ids[index]
            if progress and (event['phase'] != 'tail' or (index % 32 == 0)):
                progress({'repeat': repeat, 'row': index, **event})
    checks['repeat_exact'] = all(checks[k] for k in ('reset_exact', 'prefill_repeat_exact', 'tail_repeat_exact'))
    return saved, {'row_plan': row_plan, 'greedy_ids': greedy_ids, **checks}

def helpers():
    return {name: old.file_sha256(HERE / name) for name in
            ('gpu_qwen4_prefill_gate.py', 'gpu_qwen4_attn_split.py')}

def prepare_env(policy, shader_dir):
    require(policy in POLICIES, 'unknown policy')
    # No inherited profiling, precision, multi-device, layer or algorithm knobs.
    for key in list(os.environ):
        if key.startswith('QK_'): del os.environ[key]
    gemm, attention = POLICIES[policy]
    os.environ.update(BASE_ENV, QK_SHADER_DIR=str(Path(shader_dir).resolve()),
                      QK_FLASH_GEMM=gemm, QK_FLASH_ATTN_BATCH=attention)
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith('QK_')}

def dump(args):
    require(args.confirm_exclusive, 'use --confirm-exclusive only inside a guarded, drained Halo window')
    require(sys.byteorder == 'little' and array('f').itemsize == 4, 'requires little-endian float32 host')
    ids = old.read_ids(args.ids); row_plan = plan(len(ids), args.tail)
    require(len(ids) + 1 <= args.ctx <= 32768, 'context must cover n+1 and stay <=32768')
    path = Path(args.out); mf_path = Path(str(path) + '.json')
    require(not path.exists() and not mf_path.exists(), 'refusing to overwrite evidence')
    halo = Path('/sys/bus/pci/devices/0000:c1:00.0')
    require((halo / 'vendor').read_text().strip() == '0x1002' and
            (halo / 'device').read_text().strip() == '0x1586', 'not the expected Halo device')
    knobs = prepare_env(args.policy, args.shader_dir)
    identity = {'ids_sha256': old.file_sha256(args.ids), 'model': old.model_identity(args.model),
                'library_sha256': old.file_sha256(args.library),
                'shaders': old.shader_identity(args.shader_dir), 'helper_sha256': helpers()}
    lib = old.load_lib(args.library); engine = old.open_engine(lib, args.model, args.ctx, knobs)
    try:
        start = time.monotonic()
        payloads, checks = capture(old.EngineRunner(lib, engine), ids, args.tail,
                                  progress=lambda value: print(json.dumps({'progress': value}), flush=True))
        seconds = time.monotonic() - start
    finally:
        lib.qk_close(engine)
    unchanged = identity == {'ids_sha256': old.file_sha256(args.ids), 'model': old.model_identity(args.model),
                             'library_sha256': old.file_sha256(args.library),
                             'shaders': old.shader_identity(args.shader_dir), 'helper_sha256': helpers()}
    digest = hashlib.sha256()
    with path.open('xb') as stream:
        for payload in payloads: stream.write(payload); digest.update(payload)
    manifest = {'manifest_version': VERSION, 'experiment': 'native-prefill-exact-v1',
                'policy': args.policy, 'out': str(path.resolve()), 'dump_sha256': digest.hexdigest(),
                'ids': str(Path(args.ids).resolve()), 'n': len(ids), 'tail': args.tail,
                'ctx': args.ctx, 'vocab': VOCAB, 'chunk': CHUNK, 'knobs': knobs,
                **identity, **checks, 'seconds': seconds, 'identities_unchanged': unchanged}
    with mf_path.open('x') as stream: json.dump(manifest, stream, indent=2, allow_nan=False); stream.write('\n')
    ok = unchanged and all(checks[k] is True for k in
                          ('reset_exact', 'prefill_repeat_exact', 'tail_repeat_exact', 'repeat_exact', 'all_finite', 'abi_greedy_exact'))
    print(json.dumps({'result': 'PASS' if ok else 'FAIL', 'out': str(path), 'rows': len(row_plan),
                      'prefill_rows': sum(r['phase'] == 'prefill' for r in row_plan),
                      'dump_sha256': digest.hexdigest(), 'seconds': seconds}), flush=True)
    return 0 if ok else 1

def validate_manifest(mf):
    require(isinstance(mf, dict) and set(mf) == REQUIRED, 'incomplete/unknown manifest schema')
    require(mf['manifest_version'] == VERSION and mf['experiment'] == 'native-prefill-exact-v1', 'wrong manifest version/experiment')
    require(mf['policy'] in POLICIES, 'wrong policy')
    require(mf['chunk'] == CHUNK and type(mf['vocab']) is int and mf['vocab'] == VOCAB, 'wrong chunk/vocab')
    require(type(mf['ctx']) is int and mf['n'] + 1 <= mf['ctx'] <= 32768, 'wrong context')
    require(mf['row_plan'] == plan(mf['n'], mf['tail']), 'missing/reordered prefill or tail rows')
    for key in ('reset_exact', 'prefill_repeat_exact', 'tail_repeat_exact', 'repeat_exact',
                'all_finite', 'abi_greedy_exact', 'identities_unchanged'):
        require(mf[key] is True, f'failed {key}')
    for key in ('dump_sha256', 'ids_sha256', 'library_sha256'):
        require(valid_hash(mf[key]), f'invalid {key}')
    require(isinstance(mf['shaders'], dict) and mf['shaders'] and
            all(isinstance(k,str) and k.endswith('.spv') and valid_hash(v) for k,v in mf['shaders'].items()), 'invalid shader identity')
    require(isinstance(mf['helper_sha256'], dict) and set(mf['helper_sha256']) == set(helpers()) and
            all(valid_hash(v) for v in mf['helper_sha256'].values()), 'invalid helper identity')
    require(isinstance(mf['model'], dict) and set(mf['model']) == {'shards'} and mf['model']['shards'], 'missing model identity')
    for shard in mf['model']['shards']:
        require(set(shard) == {'path','size','mtime_ns'} and isinstance(shard['path'],str) and
                type(shard['size']) is int and shard['size'] > 0 and type(shard['mtime_ns']) is int and shard['mtime_ns'] > 0, 'invalid model shard')
    for key in ('ids', 'out'): require(isinstance(mf[key],str) and Path(mf[key]).is_absolute(), f'invalid {key}')
    require(type(mf['seconds']) in (int,float) and math.isfinite(mf['seconds']) and mf['seconds'] >= 0, 'invalid runtime')
    gemm, attn = POLICIES[mf['policy']]
    expected = {**BASE_ENV, 'QK_FLASH_GEMM': gemm, 'QK_FLASH_ATTN_BATCH': attn}
    env = mf['knobs']
    require(isinstance(env,dict) and set(env) == set(expected) | {'QK_SHADER_DIR'} and
            all(env[k] == v for k,v in expected.items()) and isinstance(env['QK_SHADER_DIR'], str) and
            Path(env['QK_SHADER_DIR']).is_absolute(), 'unmatched or ungated environment')
    require(isinstance(mf['greedy_ids'],list) and len(mf['greedy_ids']) == len(mf['row_plan']) and
            all(type(v) is int and 0 <= v < mf['vocab'] for v in mf['greedy_ids']), 'invalid greedy IDs')

def compare_paths(baseline, candidate, policy):
    require(policy in ('vec4', 'compact'), 'candidate must be vec4 or compact')
    paths = [Path(baseline), Path(candidate)]
    manifests = [json.loads(Path(str(p) + '.json').read_text()) for p in paths]
    for mf in manifests: validate_manifest(mf)
    a, b = manifests
    require((a['policy'], b['policy']) == ('baseline', policy), 'incorrect policy order')
    for key in REQUIRED - PER_RUN:
        require(a[key] == b[key], f'pair identity mismatch: {key}')
    ka, kb = dict(a['knobs']), dict(b['knobs'])
    selected = 'QK_FLASH_ATTN_BATCH' if policy == 'vec4' else 'QK_FLASH_GEMM'
    ka.pop(selected); kb.pop(selected)
    require(ka == kb, 'pair differs beyond selected policy')
    require(a['helper_sha256'] == helpers(), 'current gate helper differs from producer')
    rows = len(a['row_plan']); vocab = a['vocab']
    for path, mf in zip(paths, manifests):
        require(path.stat().st_size == rows * vocab * 4, 'wrong dump size')
        require(old.file_sha256(path) == mf['dump_sha256'], 'dump hash mismatch')
    failures = []; counts = {phase: 0 for phase in ('clean','prefill','tail')}
    with paths[0].open('rb') as fa, paths[1].open('rb') as fb:
        for index, event in enumerate(a['row_plan']):
            ba, bb = fa.read(vocab * 4), fb.read(vocab * 4)
            if ba != bb: failures.append({'row': index, **event, 'reason': 'non-bit-exact logits'})
            for label, payload, mf in (('baseline', ba, a), (policy, bb, b)):
                values = array('f'); values.frombytes(payload)
                finite, exact = row_properties(values, mf['greedy_ids'][index], vocab)
                if not finite or not exact: failures.append({'row': index, 'policy': label, 'reason': 'nonfinite/ABI argmax mismatch'})
            counts[event['phase']] += 1
    prefill = [e for e in a['row_plan'] if e['phase'] == 'prefill']
    return {'result': 'FAIL' if failures else 'PASS', 'experiment': 'native-prefill-exact-v1',
            'candidate': policy, 'bit_exact': not failures, 'saved_row_counts': counts,
            'partial_prefill_widths': [e['width'] for e in prefill if e['width'] != CHUNK],
            'n': a['n'], 'tail': a['tail'], 'ctx': a['ctx'], 'ids_sha256': a['ids_sha256'],
            'library_sha256': a['library_sha256'], 'helper_sha256': a['helper_sha256'],
            'shaders': a['shaders'], 'model': a['model'],
            'manifest_sha256': [old.file_sha256(str(p) + '.json') for p in paths],
            'dump_sha256': [mf['dump_sha256'] for mf in manifests], 'failures': failures,
            'coverage': 'clean row, last-position logits of every prefill chunk, all teacher-tail rows; complete exact reset/repeat',
            'default_promotion': False}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    d = sub.add_parser('dump'); d.add_argument('model'); d.add_argument('--policy', choices=POLICIES, required=True)
    d.add_argument('--library', required=True); d.add_argument('--shader-dir', required=True)
    d.add_argument('--ids', required=True); d.add_argument('--tail', type=int, required=True)
    d.add_argument('--ctx', type=int, default=32768); d.add_argument('--out', required=True)
    d.add_argument('--confirm-exclusive', action='store_true')
    c = sub.add_parser('compare'); c.add_argument('baseline'); c.add_argument('candidate')
    c.add_argument('--policy', required=True, choices=('vec4','compact'))
    c.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == 'dump': return dump(args)
        require(not Path(args.output).exists(), 'refusing to overwrite gate')
        result = compare_paths(args.baseline, args.candidate, args.policy)
        with Path(args.output).open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('shaders','model')}), flush=True)
        return 0 if result['result'] == 'PASS' else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'error: {error}\n')

if __name__ == '__main__': raise SystemExit(main())
