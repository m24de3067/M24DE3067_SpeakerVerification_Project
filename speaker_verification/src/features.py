"""
features.py – Audio feature extraction (Mel Filterbank & MFCC).
Wraps torchaudio transforms for consistent preprocessing.
"""
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from pathlib import Path


class FeatureExtractor(nn.Module):
    """
    Extracts log-Mel filterbank or MFCC features from raw waveforms.

    Args:
        feature_type : "fbank" | "mfcc"
        sample_rate  : expected input sample rate (Hz)
        n_mels       : number of Mel filterbank bins
        n_mfcc       : number of MFCC coefficients (used only when feature_type="mfcc")
        win_length   : FFT window length in samples
        hop_length   : FFT hop length in samples
        f_min        : minimum frequency (Hz)
        f_max        : maximum frequency (Hz)
        normalize    : apply per-utterance mean/std normalisation (CMVN)
    """

    def __init__(
        self,
        feature_type: str = "fbank",
        sample_rate:  int = 16000,
        n_mels:       int = 80,
        n_mfcc:       int = 40,
        win_length:   int = 400,
        hop_length:   int = 160,
        f_min:        float = 20.0,
        f_max:        float = 7600.0,
        normalize:    bool = True,
    ):
        super().__init__()
        self.feature_type = feature_type
        self.normalize    = normalize
        self.sample_rate  = sample_rate

        mel_kwargs = dict(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )

        if feature_type == "fbank":
            self.transform = nn.Sequential(
                T.MelSpectrogram(**mel_kwargs),
                T.AmplitudeToDB(stype="power", top_db=80),
            )
            self.out_dim = n_mels

        elif feature_type == "mfcc":
            mfcc_mel_kwargs = {k: v for k, v in mel_kwargs.items() if k != "sample_rate"}
            self.transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs=mfcc_mel_kwargs,
            )
            self.out_dim = n_mfcc

        else:
            raise ValueError(f"Unknown feature_type: {feature_type!r}")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (batch, time) or (time,)  float32, range [-1, 1]
        Returns:
            features : (batch, time_frames, out_dim)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)   # (1, time)

        # Mono: ensure single channel
        if waveform.dim() == 3:                # (batch, channels, time)
            waveform = waveform.mean(dim=1)

        feats = self.transform(waveform)       # (batch, out_dim, time_frames)
        feats = feats.transpose(1, 2)          # (batch, time_frames, out_dim)

        if self.normalize:
            mean = feats.mean(dim=1, keepdim=True)
            std  = feats.std(dim=1, keepdim=True).clamp(min=1e-5)
            feats = (feats - mean) / std

        return feats                           # (batch, time_frames, out_dim)

    @torch.no_grad()
    def extract_file(self, filepath: str) -> torch.Tensor:
        """Load a wav file and return features (1, T, D)."""
        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return self.forward(waveform)
