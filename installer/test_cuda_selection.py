"""Offline regression tests for the NVIDIA backend selection table."""
import unittest

from select_torch_backend import parse_gpu_info, select


class CudaSelectionTests(unittest.TestCase):
    def pick(self, compute, driver, index=0, gpu_name="Example GPU"):
        csv = f"{index}, {gpu_name}, GPU-test-{index}, {driver}, {compute}\n"
        result = select(parse_gpu_info(csv))
        return [x["backend"] for x in result["candidates"]], result

    def test_ampere_latest_driver(self):
        options, result = self.pick("8.6", "572.61")
        self.assertEqual(options, ["CUDA128", "CUDA126"])
        self.assertEqual(result["candidates"][0]["gpu_uuid"], "GPU-test-0")

    def test_turing_cuda126_when_driver_old(self):
        options, _ = self.pick("7.5", "561.17")
        self.assertEqual(options, ["CUDA126"])

    def test_pascal_forces_cuda126_even_with_new_driver(self):
        options, _ = self.pick("6.1", "581.80")
        self.assertEqual(options, ["CUDA126"])

    def test_volta_forces_cuda126(self):
        options, _ = self.pick("7.0", "581.80")
        self.assertEqual(options, ["CUDA126"])

    def test_blackwell_requires_cuda128(self):
        options, _ = self.pick("12.0", "581.80")
        self.assertEqual(options, ["CUDA128"])
        old_driver_options, _ = self.pick("12.0", "561.17")
        self.assertEqual(old_driver_options, [])

    def test_legacy_gpu_has_no_supported_build(self):
        options, result = self.pick("3.5", "581.80")
        self.assertEqual(options, [])
        self.assertIn("below 5.0", result["gpus"][0]["reason"])

    def test_old_driver_falls_back_cpu(self):
        options, _ = self.pick("8.6", "535.00")
        self.assertEqual(options, [])

    def test_absent_gpu_falls_back_cpu(self):
        result = select([])
        self.assertEqual(result["candidates"], [])
        self.assertTrue(result["cpu_fallback"])

    def test_multiple_gpus(self):
        result = select(parse_gpu_info(
            "0, GeForce GTX 1080, GPU-legacy, 580.50, 6.1\n"
            "1, GeForce RTX 5090, GPU-new, 580.50, 12.0\n"
        ))
        self.assertEqual(result["candidates"][0]["gpu_uuid"], "GPU-new")
        self.assertEqual(result["candidates"][0]["backend"], "CUDA128")
        self.assertEqual(result["candidates"][1]["backend"], "CUDA126")

    def test_gpu_name_with_comma(self):
        result = select(parse_gpu_info("0, GPU Special, Revision A, GPU-xyz, 580.50, 8.9\n"))
        self.assertEqual(result["gpus"][0]["name"], "GPU Special,Revision A")


if __name__ == "__main__":
    unittest.main()
