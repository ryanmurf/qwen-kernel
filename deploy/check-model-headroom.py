#!/usr/bin/env python3
"""Read-only host-memory admission check, after all model processes have exited.

MemAvailable excludes much of TTM's reusable unused pool. Do not lower this
guard just because a stopped GPU model left pages there: inspect that pool
separately. Conversely, releasing unused TTM pages does not make it safe to
load a ~90 GiB model beside a large anonymous-memory job.

Count resident anonymous/shared pages AND used swap as a conservative budget
for other jobs that can become resident again. This is not a reservation or
a guarantee against jobs started after this snapshot. It never clears swap,
shrinks GPU memory, changes limits, or stops processes.
"""
import argparse
import json
from pathlib import Path
import sys

KIB_PER_GIB = 1024**2
REQUIRED = ("MemTotal", "MemAvailable", "AnonPages", "Shmem", "SwapTotal", "SwapFree")


def parse_meminfo(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].rstrip(":") not in REQUIRED:
            continue
        key = fields[0].rstrip(":")
        if key in result or len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
            raise ValueError(f"invalid or duplicate /proc/meminfo field: {key}")
        result[key] = int(fields[1])
    if missing := set(REQUIRED) - result.keys():
        raise ValueError(f"missing /proc/meminfo fields: {', '.join(sorted(missing))}")
    if (result["MemTotal"] <= 0 or result["MemAvailable"] > result["MemTotal"]
            or result["SwapFree"] > result["SwapTotal"]
            or result["AnonPages"] + result["Shmem"] > result["MemTotal"]):
        raise ValueError("inconsistent /proc/meminfo counters")
    return result


def assess(mem, minimum_available_gib=24, maximum_other_memory_gib=12):
    for value in (minimum_available_gib, maximum_other_memory_gib):
        if type(value) is not int or not 1 <= value <= 128:
            raise ValueError("memory guard bounds must be integers in 1..128 GiB")
    swap_used = mem["SwapTotal"] - mem["SwapFree"]
    other = mem["AnonPages"] + mem["Shmem"] + swap_used
    reasons = []
    if mem["MemAvailable"] < minimum_available_gib * KIB_PER_GIB:
        reasons.append("insufficient MemAvailable; inspect host workloads and unused GPU pools")
    if other > maximum_other_memory_gib * KIB_PER_GIB:
        reasons.append("other jobs' anonymous/shared memory plus used swap exceeds the launch budget")
    return {"admitted": not reasons, "reasons": reasons,
            "available_kib": mem["MemAvailable"], "anonymous_kib": mem["AnonPages"],
            "shared_kib": mem["Shmem"], "swap_used_kib": swap_used,
            "anonymous_shared_and_swap_kib": other,
            "minimum_available_gib": minimum_available_gib,
            "maximum_other_memory_gib": maximum_other_memory_gib}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-available-gib", type=int, default=24)
    parser.add_argument("--maximum-other-memory-gib", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        result = assess(parse_meminfo(Path("/proc/meminfo").read_text()),
                        args.minimum_available_gib, args.maximum_other_memory_gib)
    except (OSError, ValueError) as error:
        print(json.dumps({"admitted": False, "error": str(error)}))
        return 2
    print(json.dumps(result))
    return 0 if result["admitted"] else 1


if __name__ == "__main__":
    sys.exit(main())
