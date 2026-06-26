import torch
from unittest.mock import patch

from src.device import select_device, select_dtype


def test_select_device_cuda():
    with patch.object(torch.cuda, "is_available", return_value=True):
        assert select_device() == "cuda:0"


def test_select_device_mps():
    with patch.object(torch.cuda, "is_available", return_value=False), \
         patch.object(torch.backends.mps, "is_available", return_value=True):
        assert select_device() == "mps"


def test_select_device_cpu():
    with patch.object(torch.cuda, "is_available", return_value=False), \
         patch.object(torch.backends.mps, "is_available", return_value=False):
        assert select_device() == "cpu"


def test_select_dtype_cuda():
    assert select_dtype("cuda:0") == torch.float16


def test_select_dtype_mps():
    assert select_dtype("mps") == torch.float32


def test_select_dtype_cpu():
    assert select_dtype("cpu") == torch.float32
