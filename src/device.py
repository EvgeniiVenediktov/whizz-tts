import torch


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str) -> torch.dtype:
    # MPS and CPU need float32; float16 is only stable on CUDA
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32
