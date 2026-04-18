"""
utils.py – Shared helpers: config loading, seeding, device, checkpoints, logging.
"""
import os
import random
import logging
import numpy as np
import torch
import yaml
from pathlib import Path


# ── Logging ────────────────────────────────────────────────────────────────
def get_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── Config ─────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Device ─────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Checkpoints ────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch: int, path: str, extra: dict = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model, optimizer, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


# ── Directory helpers ──────────────────────────────────────────────────────
def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ── Score normalisation ────────────────────────────────────────────────────
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalised dot product between two embedding batches."""
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)
