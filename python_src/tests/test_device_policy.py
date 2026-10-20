from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from experiments.paired_reference_cancel.device_policy import select_torch_device


class DevicePolicyTests(unittest.TestCase):
    def test_auto_uses_cuda_when_memory_is_sufficient(self) -> None:
        with (
            patch.dict(os.environ, {"DUBCLEAN_DEVICE": "auto"}, clear=False),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 6 * 1024**3)),
            patch("torch.cuda.get_device_name", return_value="Test GPU"),
        ):
            device, details = select_torch_device(minimum_free_gib=2.5)
        self.assertEqual(device, torch.device("cuda"))
        self.assertEqual(details["reason"], "auto_cuda")

    def test_auto_falls_back_when_gpu_is_busy(self) -> None:
        with (
            patch.dict(os.environ, {"DUBCLEAN_DEVICE": "auto"}, clear=False),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(512 * 1024**2, 6 * 1024**3)),
        ):
            device, details = select_torch_device(minimum_free_gib=2.5)
        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(details["reason"], "cuda_memory_busy")

    def test_cpu_override_is_respected(self) -> None:
        with patch.dict(os.environ, {"DUBCLEAN_DEVICE": "cpu"}, clear=False):
            device, details = select_torch_device(minimum_free_gib=0.0)
        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(details["reason"], "forced_cpu")


if __name__ == "__main__":
    unittest.main()
