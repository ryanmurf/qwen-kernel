#!/usr/bin/env python3
"""Correctness-gated, randomized Q5 GEMV geometry sweep; emits JSONL."""
import argparse
import json
import os
import random
import re
import subprocess
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="./build-halo/qk")
    p.add_argument("--device", default="STRIX_HALO")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--cold-mib", type=int, default=64)
    p.add_argument("--status-url", default="http://127.0.0.1:8091/handoff/status")
    args = p.parse_args()
    if min(args.repetitions, args.iterations) < 1 or args.cold_mib < 0:
        p.error("repetitions/iterations must be positive; cold-mib must be nonnegative")
    jobs = [(rep, kind, m, k, tpr)
            for rep in range(args.repetitions)
            for kind, m, k in [("q5_k", 640, 2560), ("q5_k", 320, 10240),
                               ("q5_1", 10240, 320), ("q5_1", 4096, 160)]
            for tpr in (4, 8, 16, 32, 64, 128, 256)]
    random.Random(1151).shuffle(jobs)
    for rep, kind, m, k, tpr in jobs:
        if args.status_url:
            with urllib.request.urlopen(args.status_url, timeout=5) as r:
                status = json.load(r)
            if status["slots"]["busy"]:
                raise SystemExit("Serving request detected; stopping sweep")
        env = {**os.environ, "QK_DEVICE_NAME": args.device,
               "QK_TPR": str(tpr), "QK_COLD_MIB": str(args.cold_mib)}
        env.pop("QK_DEVICE_PCI", None)
        env.pop("QK_DEVICE", None)
        cmd = [args.binary, kind, str(m), str(k), str(args.iterations)]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        match = re.search(r"gpu:\s+([0-9.]+) us/iter", output)
        error = re.search(r"max_rel_err = ([0-9.e+-]+)", output)
        if result.returncode or "FAIL" in output or not match or not error:
            raise RuntimeError(output)
        print(json.dumps({"device": args.device, "format": kind, "M": m, "K": k,
                          "tpr": tpr, "rep": rep, "cold_mib": args.cold_mib,
                          "gpu_us": float(match[1]), "max_rel_error": float(error[1])}), flush=True)


if __name__ == "__main__":
    main()
