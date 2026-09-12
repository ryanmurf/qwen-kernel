#!/usr/bin/env python3
"""Audit bounded last-head evidence. Never certifies full-model serving."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def require(ok, why):
    if not ok:
        raise ValueError(why)


def audit(gate, reference, profile, profile_log):
    sizes = [1, 2, 63, 64, 65, 127, 128, 129, 309, 511, 512, 513, 1024]
    for records in (gate, reference, profile):
        require(records[0]['type'] == 'metadata' and records[-1]['type'] == 'closed', 'incomplete run')
        require(records[0]['layers'] == [46, 48] and records[0]['full_model_validation'] is False, 'wrong scope')
        require(all(int(v) == 0 for _, v in (line.split() for line in
                    records[-1]['cgroup']['memory.events'].splitlines())), 'memory pressure/OOM event')
    new, old, prof = gate[0], reference[0], profile[0]
    require(not new['profile'] and not new['reference'] and old['reference'] and not old['profile']
            and prof['profile'] and not prof['reference'], 'wrong run modes')
    require(new['library_sha256'] == prof['library_sha256'], 'profile is a different library')
    require(new['harness_sha256'] == old['harness_sha256'] == prof['harness_sha256'], 'harness changed')
    require(new['shader_sha256'] == prof['shader_sha256'], 'profile shaders differ')
    require(all(old['shader_sha256'].get(k) == v for k, v in new['shader_sha256'].items()), 'reference common shader differs')
    require(new['model_shards'] == old['model_shards'] == prof['model_shards'], 'model shard metadata differs')
    def normalize(env):
        return {k: v for k, v in env.items() if k not in ('QK_SHADER_DIR', 'QK_FLASH_PROFILE')}
    require(normalize(new['environment']) == normalize(old['environment']) == normalize(prof['environment']), 'math/device environment differs')
    rows = [r for r in gate if r['type'] == 'parity']
    refs = [r for r in reference if r['type'] == 'reference']
    require([r['rows'] for r in rows] == [r['rows'] for r in refs] == sizes, 'incomplete parity cases')
    for row, ref in zip(rows, refs):
        require(row['bit_exact'] is True and row['compared_logit_rows'] == 4
                and row['continuation_sizes'] == [1, 2, 63], 'incomplete same-library gate')
        require(row['sha256'] == ref['sha256'] and row['all_ids_sha256'] == ref['all_ids_sha256'], 'old ABI regression')
        require(len(row['sha256']) == len(row['all_ids_sha256']) == 4, 'missing hash')
        require(all(re.fullmatch('[0-9a-f]{64}', h) for h in row['sha256'] + row['all_ids_sha256']), 'invalid hash')
    require(len([r for r in gate if r['type'] == 'result' and r['result'] == 'PASS']) == 1, 'no unique PASS')
    require(any(r['type'] == 'validation' and r['bad_args_rejected'] and r['zero_input_exact'] for r in gate), 'validation missing')
    require(any(r['type'] == 'reference_complete' and r['rows'] == 13 for r in reference), 'reference incomplete')
    timings = [r for r in gate if r['type'] == 'timing']
    require([r['rows'] for r in timings] == [128, 512, 1024], 'incomplete timing sizes')
    summary = []
    for row in timings:
        samples = [r for r in gate if r['type'] == 'sample' and r['rows'] == row['rows']]
        require([r['last_only'] for r in samples] == [False, True, True, False] * 3, 'not balanced ABBA')
        require(len({r['final_id'] for r in samples}) == 1, 'benchmark final ID differs')
        for flag, key in ((False, 'all'), (True, 'last')):
            values = [r['seconds'] for r in samples if r['last_only'] == flag]
            require(values == row[key + '_seconds'] and all(v > 0 for v in values), 'sample mismatch')
            require(statistics.median(values) == row[key + '_median'], 'wrong median')
        require(row['speedup'] == row['all_median'] / row['last_median'], 'wrong ratio')
        summary.append({k: row[k] for k in ('rows', 'all_median', 'last_median', 'speedup')})
    endings = [r for r in profile if r['type'] == 'profile_end']
    require([r['last_only'] for r in endings] == [False, True] and len({r['final_id'] for r in endings}) == 1, 'profile incomplete')
    blocks = re.split(r'\[last-head profile\] last_only=(False|True) rows=512\n', profile_log)
    require(len(blocks) == 5 and blocks[1] == 'False' and blocks[3] == 'True', 'profile markers missing')
    head = []
    for flag, block in ((False, blocks[2]), (True, blocks[4])):
        dispatches = re.findall(r'\[flash raw\]\s+\d+\s+(\S+)\s+([0-9.]+) us', block)
        indices = [i for i, (name, _) in enumerate(dispatches) if name == 'qwen4_argmax.spv']
        require(len(indices) == (1 if flag else 8), 'wrong number of vocabulary tiles')
        require(all(i > 0 and dispatches[i-1][0] == 'qwen4_gemm_q6k.spv' for i in indices), 'head arithmetic changed')
        head.append(dict(last_only=flag, vocabulary_tiles=len(indices),
                         vocabulary_and_argmax_ms=sum(float(dispatches[i-j][1]) for i in indices for j in (0, 1))/1000))
    return dict(result='PASS', full_model_validation=False, default_promotion=False,
                parity_sizes=sizes, exact_final_logit_rows=52, old_abi_exact_id_buffers=52,
                reference_extra_shaders=sorted(old['shader_sha256'].keys() - new['shader_sha256'].keys()),
                library_sha256=new['library_sha256'], reference_library_sha256=old['library_sha256'],
                harness_sha256=new['harness_sha256'], timings_partial_stage_only=summary,
                profile_diagnostic_only=head)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gate', 'reference', 'profile', 'profile_log'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    inputs = [args.gate, args.reference, args.profile, args.profile_log]
    data = [p.read_bytes() for p in inputs]
    try:
        result = audit(*[[json.loads(line) for line in raw.decode().splitlines()] for raw in data[:3]], data[3].decode())
        result['evidence_sha256'] = {p.name: hashlib.sha256(raw).hexdigest() for p, raw in zip(inputs, data)}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'audit failed: {error}\n')


if __name__ == '__main__':
    main()
