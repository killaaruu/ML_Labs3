"""Общие утилиты для лабораторных работ."""
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_amp(device: torch.device, enabled: bool = True):
    """Смешанная точность (fp16 на тензорных ядрах) — только на CUDA; на MPS/CPU всё как обычно.

    Возвращает функцию-контекст autocast() и GradScaler (при выключенном AMP он ничего не делает).
    """
    on = enabled and device.type == "cuda"
    if on:
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=on)

    def autocast():
        return torch.autocast("cuda", dtype=torch.float16) if on else nullcontext()

    return autocast, scaler