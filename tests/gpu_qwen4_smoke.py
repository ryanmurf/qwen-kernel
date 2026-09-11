#!/usr/bin/env python3
"""Explicit, idle-guarded GPU operator checks; no model-sized allocations."""
import argparse
import json
import os
import subprocess
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="./build-halo/qk")
    p.add_argument("--device", default="STRIX_HALO")
    p.add_argument("--status-url", default="http://127.0.0.1:8091/handoff/status")
    args = p.parse_args()
    env = {**os.environ, "QK_DEVICE_NAME": args.device, "QK_COLD_MIB": "0"}
    for key in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_TPR"):
        env.pop(key, None)
    jobs = [([kind, str(m), str(k), "3"], tpr)
            for kind, m, k in (("q5_k", 7, 256), ("q5_k", 65, 2560),
                               ("q5_1", 7, 32), ("q5_1", 67, 160), ("q5_1", 65, 320))
            for tpr in (4, 8, 16, 32, 64, 128, 256)]
    jobs += [(["qwen4-hc", str(n), str(h), str(t)], None)
             for n, h, t in ((1, 1, 1), (65, 4, 3), (2560, 4, 1),
                              (2560, 4, 8), (257, 3, 2), (2560, 4, 32))]
    for command, tpr in jobs:
        if args.status_url:
            with urllib.request.urlopen(args.status_url, timeout=5) as r:
                if json.load(r)["slots"]["busy"]:
                    raise SystemExit("Serving request active; stopping operator tests")
        run_env = {**env, **({"QK_TPR": str(tpr)} if tpr else {})}
        result = subprocess.run([args.binary, *command], env=run_env, capture_output=True,
                                text=True, timeout=120)
        output = result.stdout + result.stderr
        assert result.returncode == 0 and "PASS" in output and "FAIL" not in output, output
        print(json.dumps({"device": args.device, "args": command, "tpr": tpr,
                          "passed": True}), flush=True)


if __name__ == "__main__":
    main()
