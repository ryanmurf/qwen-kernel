#!/usr/bin/env python3
"""Actual-GGUF Q5 MoE geometry sweep. Allocates one layer's weights, not the model."""
import argparse
import json
import os
import random
import re
import subprocess
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--binary", default="./build-halo/qk")
    p.add_argument("--device", default="STRIX_HALO")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--status-url", default="http://127.0.0.1:8091/handoff/status")
    args = p.parse_args()
    if args.layer < 0 or min(args.repetitions, args.iterations) < 1:
        p.error("layer must be nonnegative; repetitions/iterations must be positive")
    jobs = [(rep, gate, down) for rep in range(args.repetitions)
            for gate in (64, 128, 256) for down in (64, 128, 256)]
    random.Random(397).shuffle(jobs)
    for rep, gate, down in jobs:
        if args.status_url:
            with urllib.request.urlopen(args.status_url, timeout=5) as response:
                if json.load(response)["slots"]["busy"]:
                    raise SystemExit("Serving request active; stopping MoE sweep")
        env = {**os.environ, "QK_GGUF": args.model, "QK_DEVICE_NAME": args.device,
               "QK_MOE_Q5_WG": str(gate), "QK_MOE_Q8_WG": str(down)}
        for key in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_TPR", "QK_IMPORT_WEIGHTS"):
            env.pop(key, None)
        result = subprocess.run([args.binary, "moe", str(args.layer), str(args.iterations)],
                                env=env, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        timing = re.search(r"gpu:\s+([0-9.]+) µs/layer-moe", output)
        error = re.search(r"max_rel_err = ([0-9.e+-]+)", output)
        if result.returncode or "FAIL" in output or "MATCH" not in output or not timing or not error:
            raise RuntimeError(output)
        print(json.dumps({"device": args.device, "layer": args.layer, "rep": rep,
                          "gate_wg": gate, "down_wg": down,
                          "gpu_us": float(timing[1]), "max_rel_error": float(error[1])}), flush=True)


if __name__ == "__main__":
    main()
