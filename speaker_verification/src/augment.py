"""
augment.py – Audio augmentation: additive noise, reverberation, speed perturbation.
"""
import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Optional


class AudioAugmenter:
    """
    On-the-fly audio augmentation pipeline.

    Augmentations applied (each with independent probability):
      1. Additive noise from MUSAN at random SNR
      2. Convolutive reverberation via RIR
      3. Speed perturbation (torchaudio SoX effect)

    Falls back to identity (no augmentation) when files are unavailable,
    so the pipeline runs without the MUSAN corpus for dry-run testing.
    """

    def __init__(
        self,
        musan_path:    Optional[str] = None,
        rir_path:      Optional[str] = None,
        sample_rate:   int   = 16000,
        noise_prob:    float = 0.5,
        reverb_prob:   float = 0.3,
        speed_prob:    float = 0.3,
        snr_low_db:    float = 5.0,
        snr_high_db:   float = 20.0,
        speed_factors: List[float] = (0.9, 1.0, 1.1),
    ):
        self.sample_rate   = sample_rate
        self.noise_prob    = noise_prob
        self.reverb_prob   = reverb_prob
        self.speed_prob    = speed_prob
        self.snr_low_db    = snr_low_db
        self.snr_high_db   = snr_high_db
        self.speed_factors = list(speed_factors)

        self.noise_files: List[Path] = []
        self.rir_files:   List[Path] = []

        if musan_path and Path(musan_path).exists():
            self.noise_files = sorted(Path(musan_path).rglob("*.wav"))

        if rir_path and Path(rir_path).exists():
            self.rir_files = sorted(Path(rir_path).rglob("*.wav"))

    # ── public API ────────────────────────────────────────────────────────

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a 1-D waveform tensor."""
        if random.random() < self.noise_prob:
            waveform = self._add_noise(waveform)
        if random.random() < self.reverb_prob:
            waveform = self._add_reverb(waveform)
        if random.random() < self.speed_prob:
            waveform = self._speed_perturb(waveform)
        return waveform

    def augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Augment a batch of waveforms (batch, time)."""
        return torch.stack([self(wav) for wav in batch])

    # ── internal helpers ──────────────────────────────────────────────────

    def _load_random(self, file_list: List[Path]) -> Optional[torch.Tensor]:
        if not file_list:
            return None
        path = random.choice(file_list)
        try:
            wav, sr = torchaudio.load(str(path))
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            return wav.mean(dim=0)   # mono (time,)
        except Exception:
            return None

    def _add_noise(self, signal: torch.Tensor) -> torch.Tensor:
        noise = self._load_random(self.noise_files)
        if noise is None:
            # Synthesise Gaussian noise if no corpus available
            noise = torch.randn_like(signal) * 0.01

        # Match length
        if noise.shape[0] < signal.shape[0]:
            reps  = (signal.shape[0] // noise.shape[0]) + 1
            noise = noise.repeat(reps)
        start = random.randint(0, noise.shape[0] - signal.shape[0])
        noise = noise[start: start + signal.shape[0]]

        snr_db      = random.uniform(self.snr_low_db, self.snr_high_db)
        sig_rms     = signal.norm(p=2).clamp(min=1e-9)
        noise_rms   = noise.norm(p=2).clamp(min=1e-9)
        scale       = sig_rms / (noise_rms * (10 ** (snr_db / 20)))
        return signal + scale * noise

    def _add_reverb(self, signal: torch.Tensor) -> torch.Tensor:
        rir = self._load_random(self.rir_files)
        if rir is None:
            return signal
        try:
            reverbed = torchaudio.functional.fftconvolve(
                signal.unsqueeze(0), rir.unsqueeze(0)
            ).squeeze(0)
            # Trim to original length
            return reverbed[: signal.shape[0]]
        except Exception:
            return signal

    def _speed_perturb(self, signal: torch.Tensor) -> torch.Tensor:
        factor = random.choice(self.speed_factors)
        if factor == 1.0:
            return signal
        try:
            effects   = [["speed", str(factor)], ["rate", str(self.sample_rate)]]
            perturbed, _ = torchaudio.sox_effects.apply_effects_tensor(
                signal.unsqueeze(0), self.sample_rate, effects
            )
            return perturbed.squeeze(0)[: signal.shape[0]]
        except Exception:
            return signal


# ── Convenience: fixed-noise augmenter for reproducible tests ─────────────

class GaussianNoiseAugmenter:
    """Lightweight augmenter using only Gaussian noise (no corpus needed)."""

    def __init__(self, noise_std: float = 0.005):
        self.noise_std = noise_std

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform + torch.randn_like(waveform) * self.noise_std
