#!/usr/bin/env python3
"""Compare a bounded native prefix against an independently captured F32 oracle.

This is not a full-model or serving test. The native process loads approximately
2 GiB per layer (maximum four); do not run it on a fully occupied device.
"""
import argparse
import json
import os
import re
import subprocess
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("reference")
    p.add_argument("--last-layer", type=int, choices=range(4), default=1)
    p.add_argument("--token", type=int, default=198)
    p.add_argument("--steps", type=int, choices=range(1, 33), default=16)
    p.add_argument("--binary", default="./build-halo/qk")
    p.add_argument("--device", default="STRIX_HALO")
    p.add_argument("--status-url", default="http://127.0.0.1:8091/handoff/status")
    args = p.parse_args()
    if args.token < 0 or not os.path.isfile(args.reference) or os.path.getsize(args.reference) != args.steps * 10240 * 4:
        p.error("invalid token or reference size; wait for the oracle to finish")
    if args.status_url:
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            if json.load(response)["slots"]["busy"]:
                raise SystemExit("Serving request active; stopping prefix test")
    env = {**os.environ, "QK_DEVICE_NAME": args.device, "QK_GGUF": args.model}
    for name in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_TPR", "QK_LAYER_DUMP"):
        env.pop(name, None)
    command = [args.binary, "qwen4-prefix", str(args.last_layer), str(args.token),
               args.reference, str(args.steps)]
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    output = run.stdout + run.stderr
    error = re.search(r"relative_rms=([0-9.e+-]+) max_abs=([0-9.e+-]+)", output)
    frame = re.search(r"worst frame relative_rms=([0-9.e+-]+); reset exact", output)
    if run.returncode or not error or not frame or "-> PASS" not in output or "FAIL" in output:
        raise RuntimeError(output)
    print(json.dumps({"device": args.device, "layers": args.last_layer + 1,
                      "token_start": args.token, "steps": args.steps,
                      "relative_rms": float(error[1]), "max_abs": float(error[2]),
                      "worst_frame_relative_rms": float(frame[1]), "reset_exact": True,
                      "scope": "native prefix; not logits or serving"}))


if __name__ == "__main__":
    main()
