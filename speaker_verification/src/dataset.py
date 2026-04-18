"""
dataset.py – PyTorch Dataset classes for VoxCeleb and demographic evaluation.
"""
import os
import random
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict


# ══════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════

def load_audio(path: str, target_sr: int = 16000, max_duration: float = 10.0) -> torch.Tensor:
    """Load and resample a waveform; clip to max_duration seconds."""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.mean(dim=0)                              # mono
    max_samples = int(max_duration * target_sr)
    if wav.shape[0] > max_samples:
        start = random.randint(0, wav.shape[0] - max_samples)
        wav   = wav[start: start + max_samples]
    return wav                                         # (time,)


def pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    """Pad with zeros or randomly crop to fixed length."""
    if wav.shape[0] >= length:
        start = random.randint(0, wav.shape[0] - length)
        return wav[start: start + length]
    return torch.nn.functional.pad(wav, (0, length - wav.shape[0]))


# ══════════════════════════════════════════════════════════════════════════
# VoxCeleb Training Dataset
# ══════════════════════════════════════════════════════════════════════════

class VoxCelebDataset(Dataset):
    """
    Training dataset for VoxCeleb1 / VoxCeleb2.

    Expects a CSV file with columns: [path, speaker_id, gender, nationality]
    where speaker_id is an integer label (0-indexed).

    If no real data is found, generates synthetic waveforms so the
    pipeline can be tested end-to-end without downloading the corpus.
    """

    FIXED_LEN = 32000   # 2 seconds @ 16 kHz

    def __init__(
        self,
        csv_path:       str,
        sample_rate:    int   = 16000,
        augmenter=None,
        max_duration:   float = 10.0,
        synthetic:      bool  = False,
    ):
        self.sample_rate  = sample_rate
        self.augmenter    = augmenter
        self.max_duration = max_duration
        self.synthetic    = synthetic

        if synthetic or not Path(csv_path).exists():
            self.df = self._make_synthetic_df()
        else:
            self.df = pd.read_csv(csv_path)
            self._validate_columns()

        self.speaker2idx = {s: i for i, s in enumerate(self.df["speaker_id"].unique())}
        self.n_speakers  = len(self.speaker2idx)

    # ── helpers ───────────────────────────────────────────────────────────
    def _validate_columns(self):
        required = {"path", "speaker_id"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

    @staticmethod
    def _make_synthetic_df(n_speakers: int = 50, utts_per_spk: int = 20) -> pd.DataFrame:
        """Return a DataFrame with synthetic metadata (no real audio)."""
        genders = ["male", "female"]
        nats    = ["native", "non_native"]
        rows    = []
        for spk in range(n_speakers):
            for utt in range(utts_per_spk):
                rows.append({
                    "path":        f"synthetic/{spk:04d}/{utt:04d}.wav",
                    "speaker_id":  spk,
                    "gender":      genders[spk % 2],
                    "nationality": nats[spk % 2],
                })
        return pd.DataFrame(rows)

    # ── Dataset interface ─────────────────────────────────────────────────
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row        = self.df.iloc[idx]
        label      = self.speaker2idx[row["speaker_id"]]

        if self.synthetic or not Path(row["path"]).exists():
            wav = torch.randn(self.FIXED_LEN) * 0.05
        else:
            wav = load_audio(row["path"], self.sample_rate, self.max_duration)

        wav = pad_or_trim(wav, self.FIXED_LEN)

        if self.augmenter is not None:
            wav = self.augmenter(wav)

        return wav, label


# ══════════════════════════════════════════════════════════════════════════
# Verification (Trial) Dataset  — pairs of utterances + same/diff label
# ══════════════════════════════════════════════════════════════════════════

class VerificationDataset(Dataset):
    """
    Dataset for speaker verification evaluation.
    Each item is (wav1, wav2, label) where label=1 means same speaker.

    CSV columns: [path1, path2, label, gender1, gender2, nationality1, nationality2]
    """

    FIXED_LEN = 48000   # 3 s

    def __init__(
        self,
        csv_path:    str,
        sample_rate: int = 16000,
        synthetic:   bool = False,
    ):
        self.sample_rate = sample_rate
        self.synthetic   = synthetic

        if synthetic or not Path(csv_path).exists():
            self.df = self._make_synthetic_trials(n_trials=500)
        else:
            self.df = pd.read_csv(csv_path)

    @staticmethod
    def _make_synthetic_trials(n_trials: int = 500) -> pd.DataFrame:
        rows = []
        genders = ["male", "female"]
        nats    = ["native", "non_native"]
        for i in range(n_trials):
            same = random.random() > 0.5
            rows.append({
                "path1":       f"synthetic/spk{i%50}/utt{i%10}.wav",
                "path2":       f"synthetic/spk{i%50}/utt{(i+1)%10}.wav" if same
                               else f"synthetic/spk{(i+1)%50}/utt0.wav",
                "label":       int(same),
                "gender1":     genders[i % 2],
                "gender2":     genders[i % 2],
                "nationality1": nats[i % 2],
                "nationality2": nats[(i+1) % 2],
            })
        return pd.DataFrame(rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        def _load(path):
            if self.synthetic or not Path(path).exists():
                return torch.randn(self.FIXED_LEN) * 0.05
            wav = load_audio(path, self.sample_rate)
            return pad_or_trim(wav, self.FIXED_LEN)

        wav1  = _load(row["path1"])
        wav2  = _load(row["path2"])
        label = int(row["label"])

        meta = {
            "gender1":      row.get("gender1",      "unknown"),
            "gender2":      row.get("gender2",      "unknown"),
            "nationality1": row.get("nationality1", "unknown"),
            "nationality2": row.get("nationality2", "unknown"),
        }
        return wav1, wav2, label, meta


# ══════════════════════════════════════════════════════════════════════════
# Demographic Evaluation Wrapper
# ══════════════════════════════════════════════════════════════════════════

class DemoDataset:
    """
    Splits a VerificationDataset into per-demographic subsets.
    Returns dict: {group_name: list of (score, label)} after model inference.
    """

    GROUPS = {
        "male":       lambda m: m["gender1"] == "male"    and m["gender2"] == "male",
        "female":     lambda m: m["gender1"] == "female"  and m["gender2"] == "female",
        "native":     lambda m: m["nationality1"] == "native",
        "non_native": lambda m: m["nationality1"] == "non_native",
    }

    @staticmethod
    def split(dataset: VerificationDataset) -> Dict[str, List[int]]:
        """Returns {group: [indices]} for per-group evaluation."""
        splits = {g: [] for g in DemoDataset.GROUPS}
        for i in range(len(dataset)):
            _, _, _, meta = dataset[i]
            for group, predicate in DemoDataset.GROUPS.items():
                if predicate(meta):
                    splits[group].append(i)
        return splits
