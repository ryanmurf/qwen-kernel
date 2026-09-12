"""CPU-only checks: no real memory maintenance, process control, or GPU use."""
import importlib.util
import io
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("headroom",
    Path(__file__).resolve().parents[1] / "deploy" / "check-model-headroom.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
GIB = 1024**2


def meminfo(**overrides):
    values = dict(MemTotal=121*GIB, MemAvailable=30*GIB, AnonPages=GIB,
                  Shmem=0, SwapTotal=16*GIB, SwapFree=14*GIB)
    values.update(overrides)
    return "\n".join(f"{key}: {value} kB" for key, value in values.items())


class HeadroomTests(unittest.TestCase):
    def assess(self, **overrides):
        return mod.assess(mod.parse_meminfo(meminfo(**overrides)))

    def test_healthy_stopped_model_host(self):
        self.assertTrue(self.assess()["admitted"])

    def test_low_available_refuses(self):
        self.assertFalse(self.assess(MemAvailable=23*GIB)["admitted"])

    def test_released_gpu_pool_does_not_hide_large_other_job(self):
        self.assertFalse(self.assess(MemAvailable=50*GIB, AnonPages=24*GIB)["admitted"])

    def test_swapped_working_set_is_not_ignored(self):
        rec = self.assess(SwapFree=4*GIB)
        self.assertFalse(rec["admitted"])
        self.assertEqual(rec["anonymous_shared_and_swap_kib"], 13*GIB)

    def test_shared_memory_counts(self):
        self.assertFalse(self.assess(Shmem=10*GIB)["admitted"])

    def test_exact_thresholds(self):
        self.assertTrue(self.assess(MemAvailable=24*GIB, AnonPages=10*GIB)["admitted"])

    def test_malformed_or_inconsistent_counters_refuse(self):
        bad = [meminfo() + "\nMemTotal: 5 kB", meminfo().replace("AnonPages:", "Missing:"),
               meminfo().replace(" kB", " bytes"), meminfo(Shmem=-1),
               meminfo(MemAvailable=122*GIB), meminfo(SwapFree=17*GIB),
               meminfo(AnonPages=120*GIB, Shmem=2*GIB)]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                mod.parse_meminfo(text)

    def test_invalid_bounds(self):
        for value in (0, 129, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mod.assess(mod.parse_meminfo(meminfo()), value, 12)

    def test_cli_exit_codes(self):
        for text, expected in [(meminfo(), 0), (meminfo(AnonPages=24*GIB), 1), ("broken", 2)]:
            with patch.object(mod.Path, "read_text", return_value=text), redirect_stdout(io.StringIO()):
                self.assertEqual(mod.main([]), expected)


if __name__ == "__main__":
    unittest.main()
