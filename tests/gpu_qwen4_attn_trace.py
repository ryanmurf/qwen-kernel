#!/usr/bin/env python3
"""Capture existing layer taps around a long-run divergence on the Halo only.

This is a diagnostic, not a performance or accuracy gate. It uses a prior
complete dump as its replay oracle and requires the instrumented logits to
match that dump exactly before allowing layer-level attribution. Outputs
are confined to a newly created directory; existing traces are not replaced.
"""
import argparse
from array import array
import importlib.util
import json
import os
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('attn_split', Path(__file__).with_name('gpu_qwen4_attn_split.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--mode', required=True, choices=('serial', 'split'))
    parser.add_argument('--reference', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--first-tail-index', type=int, default=48)
    parser.add_argument('--last-tail-index', type=int, default=49)
    parser.add_argument('--library', default='build-halo/libqk.so')
    args = parser.parse_args()
    reference = Path(args.reference).resolve()
    mf = json.loads(Path(str(reference) + '.json').read_text())
    if not (0 <= args.first_tail_index <= args.last_tail_index < mf['tail']):
        raise RuntimeError('capture range is outside the recorded tail')
    if (mf['mode'] != args.mode or not mf['reset_exact'] or not mf['repeat_exact']
            or mf['library_sha256'] != helper.file_sha256(args.library)
            or mf['model'] != helper.model_identity(args.model)
            or mf['dump_sha256'] != helper.file_sha256(reference)
            or mf['ids_sha256'] != helper.file_sha256(mf['ids'])):
        raise RuntimeError('reference identity or reset/repeat check mismatch')
    ids = helper.read_ids(mf['ids'])
    if len(ids) != mf['n'] or mf['vocab'] != helper.VOCAB:
        raise RuntimeError('wrong sequence or vocabulary size')
    knobs = helper.prepare_env('STRIX_HALO', args.mode)
    if knobs != mf['knobs'] or helper.shader_identity(knobs['QK_SHADER_DIR']) != mf['shaders']:
        raise RuntimeError('runtime knobs or shaders differ from the reference')
    out = Path(args.out_dir).resolve()
    out.mkdir(mode=0o700)  # Exclusive output root; engine tap writes cannot replace old results.
    save = lambda name, value: helper.write_exclusive(str(out / name), json.dumps(value, indent=2).encode())
    save('identity.json', {'reference': str(reference), 'reference_sha256': mf['dump_sha256'],
                         'library_sha256': mf['library_sha256'], 'mode': args.mode, 'knobs': knobs,
                         'first_tail_index': args.first_tail_index, 'last_tail_index': args.last_tail_index})
    lib = helper.load_lib(args.library)
    engine = helper.open_engine(lib, args.model, mf['ctx'], knobs)
    checks = []
    try:
        runner = helper.EngineRunner(lib, engine)
        runner.step([ids[0]], 0)
        with reference.open('rb') as f:
            clean = array('f'); clean.frombytes(f.read(helper.VOCAB * 4))
        if runner.row() != clean:
            raise RuntimeError('clean row differs before tracing')
        prefix = len(ids) - mf['tail']
        for pos in range(0, prefix, helper.CHUNK):
            runner.step(ids[pos:min(pos + helper.CHUNK, prefix)], pos)
            print(json.dumps({'prefilled': min(pos + helper.CHUNK, prefix), 'prefix': prefix}), flush=True)
        with reference.open('rb') as f:
            f.seek(helper.VOCAB * 4)
            for index in range(args.last_tail_index + 1):
                if index >= args.first_tail_index:
                    os.environ['QK_LAYER_DUMP'] = str(out / f'tail-{index}')
                position = prefix + index
                greedy = runner.step([ids[position]], position)
                row = runner.row()
                expected = array('f'); expected.frombytes(f.read(helper.VOCAB * 4))
                exact = row == expected
                checks.append({'tail_index': index, 'position': position, 'reference_exact': exact,
                               'greedy': greedy, 'expected_greedy': mf['greedy_after_tail'][index],
                               'taps_enabled': index >= args.first_tail_index})
                if index >= args.first_tail_index:
                    helper.write_exclusive(str(out / f'tail-{index}.logits.f32'), bytes(row))
                if not exact or greedy != mf['greedy_after_tail'][index]:
                    raise RuntimeError(f'trace differs from its uninstrumented reference at tail {index}; do not attribute layer differences')
    finally:
        os.environ.pop('QK_LAYER_DUMP', None)
        lib.qk_close(engine)
        save('replay-checks.json', checks)
    files = sorted(out.glob('tail-*'))
    save('complete.json', {'result': 'REFERENCE-EXACT', 'checks': len(checks),
                           'files': {p.name: {'size': p.stat().st_size, 'sha256': helper.file_sha256(p)} for p in files}})
    print(json.dumps({'result': 'REFERENCE-EXACT', 'out_dir': str(out), 'captured_files': len(files)}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
