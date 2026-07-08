#!/usr/bin/env python3
"""Backend benchmark through the claude CLI (claude-qwen wrapper).
Usage: qbench.py <label> <outdir>
Writes <outdir>/results-<label>.jsonl, one record per CLI call.
"""
import json, subprocess, sys, time, glob

LABEL, OUT = sys.argv[1], sys.argv[2]
JL = f"{OUT}/results-{LABEL}.jsonl"

CODE = '''
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    out = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out

def busiest_window(events, width):
    events.sort()
    best, count, lo = 0, 0, 0
    for hi, t in enumerate(events):
        count += 1
        while events[lo] < t - width:
            lo += 1
            count -= 1
        best = max(best, count)
    return best
'''

def junction_mc():
    for p in glob.glob('/sys/class/drm/card0/device/hwmon/hwmon*/temp2_input'):
        try:
            return int(open(p).read().strip())
        except OSError:
            pass
    return None

def call(scen, it, turn, prompt, resume=None):
    cmd = ['claude-qwen', '-p', '--output-format', 'json']
    if resume:
        cmd += ['--resume', resume]
    cmd.append(prompt)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    rec = {'label': LABEL, 'scenario': scen, 'iter': it, 'turn': turn,
           'wall_s': round(wall, 2), 'rc': r.returncode, 'junc_mC': junction_mc()}
    sid = None
    try:
        j = json.loads(r.stdout)
        u = j.get('usage', {})
        rec.update({
            'api_ms': j.get('duration_api_ms'), 'dur_ms': j.get('duration_ms'),
            'num_turns': j.get('num_turns'),
            'in_tok': u.get('input_tokens'), 'out_tok': u.get('output_tokens'),
            'cache_read': u.get('cache_read_input_tokens'),
            'is_error': j.get('is_error'),
            'result_head': (j.get('result') or '')[:120]})
        sid = j.get('session_id')
    except (json.JSONDecodeError, TypeError):
        rec['stderr'] = r.stderr[-400:]
        rec['stdout_head'] = r.stdout[:200]
    with open(JL, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f"[{LABEL}] {scen}#{it}.{turn} rc={r.returncode} wall={wall:.1f}s "
          f"api={rec.get('api_ms')}ms out={rec.get('out_tok')} turns={rec.get('num_turns')}",
          flush=True)
    return sid

# S0 ping x3 — fixed overhead / TTFT proxy
for i in range(3):
    call('S0-ping', i, 0, 'Reply with exactly one word: pong')

# S1 generate x3 — sustained decode
for i in range(3):
    call('S1-generate', i, 0,
         'Explain in about 250 words how a hash map works: buckets, hashing, '
         'collisions, resizing. Plain prose, no code, no lists.')

# S2 tool call x3 — agentic loop
for i in range(3):
    call('S2-tool', i, 0,
         "Use the Bash tool to run exactly this command: wc -l /etc/hosts . "
         "Then tell me only the line count number.")

# S3 multi-turn x2 chains — resumed session, growing context
TURNS = [
    'Here is some Python code:\n```python\n' + CODE + '\n```\n'
    'Explain in 3 sentences what each function does.',
    'List three edge cases or bugs in busiest_window. Be brief.',
    'Show a fixed version of busiest_window only. Just the code.',
    'Write one pytest test function that would have caught the main bug. Just the code.',
]
for i in range(2):
    sid = None
    for t, prompt in enumerate(TURNS):
        sid = call('S3-multiturn', i, t, prompt, resume=sid) or sid

print('done', JL)
