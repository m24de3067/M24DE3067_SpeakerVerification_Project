"""
model.py – Speaker embedding models.

Implements:
  • XVectorModel   – TDNN x-vector with statistics pooling
  • ECAPAModel     – Simplified ECAPA-TDNN with channel-attention pooling
Both expose get_embedding() for downstream scoring.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════════════════════════════════════

class TDNNLayer(nn.Module):
    """1-D Time-Delay Neural Network layer with dilation."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size,
                              dilation=dilation, padding=padding)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.act  = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class StatsPooling(nn.Module):
    """Temporal statistics pooling: mean || std → 2× channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=2)
        std  = (x.var(dim=2) + 1e-9).sqrt()
        return torch.cat([mean, std], dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class Res2Conv1d(nn.Module):
    """Res2Net-style multi-scale conv block used in ECAPA."""

    def __init__(self, channels: int, scale: int = 8, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale    = scale
        self.width    = channels // scale
        padding       = (kernel_size - 1) * dilation // 2
        self.convs    = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size,
                      dilation=dilation, padding=padding)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spx = torch.split(x, self.width, dim=1)
        out = [spx[0]]
        sp  = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            sp = sp + spx[i + 1] if i > 0 else spx[i + 1]
            sp = F.relu(bn(conv(sp)))
            out.append(sp)
        return torch.cat(out, dim=1)


# ══════════════════════════════════════════════════════════════════════════
# X-Vector Model
# ══════════════════════════════════════════════════════════════════════════

class XVectorModel(nn.Module):
    """
    Standard x-vector extractor: 5-layer TDNN → stats pooling → 2 FC → classifier.

    Reference: Snyder et al., "X-Vectors: Robust DNN Embeddings for Speaker
               Recognition", ICASSP 2018.
    """

    def __init__(
        self,
        input_dim:     int = 80,
        embedding_dim: int = 512,
        num_speakers:  int = 1211,
        dropout:       float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.frame_layers = nn.Sequential(
            TDNNLayer(input_dim, 512, kernel_size=5, dilation=1),
            TDNNLayer(512,        512, kernel_size=3, dilation=2),
            TDNNLayer(512,        512, kernel_size=3, dilation=3),
            TDNNLayer(512,        512, kernel_size=1, dilation=1),
            TDNNLayer(512,       1500, kernel_size=1, dilation=1),
        )

        self.stats_pool = StatsPooling()  # 1500 → 3000

        self.segment1 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embedding_layer = nn.Linear(512, embedding_dim)
        self.classifier       = nn.Linear(embedding_dim, num_speakers)

    def get_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (batch, time_frames, input_dim)
        Returns:
            embeddings : (batch, embedding_dim)
        """
        x = features.transpose(1, 2)         # (B, D, T) for Conv1d
        x = self.frame_layers(x)
        x = self.stats_pool(x)               # (B, 3000)
        x = self.segment1(x)
        return self.embedding_layer(x)       # (B, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns speaker logits for training with CrossEntropyLoss."""
        return self.classifier(self.get_embedding(features))


# ══════════════════════════════════════════════════════════════════════════
# ECAPA-TDNN (Simplified)
# ══════════════════════════════════════════════════════════════════════════

class ECAPABlock(nn.Module):
    """One ECAPA-TDNN block: Res2Conv → BN → ReLU → SE → residual."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, scale: int = 8):
        super().__init__()
        padding    = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.res2  = Res2Conv1d(channels, scale=scale, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.bn    = nn.BatchNorm1d(channels)
        self.se    = SEBlock(channels)
        self.act   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn(self.conv1(x)))
        x = self.res2(x)
        x = self.act(self.bn(self.conv2(x)))
        x = self.se(x)
        return x + residual


class AttentiveStatsPooling(nn.Module):
    """Channel-dependent attentive statistics pooling (ECAPA-TDNN)."""

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        global_mean = x.mean(dim=2, keepdim=True).expand_as(x)
        global_std  = (x.var(dim=2, keepdim=True) + 1e-9).sqrt().expand_as(x)
        attn_input  = torch.cat([x, global_mean, global_std], dim=1)
        alpha       = self.attention(attn_input)               # (B, C, T)
        mean        = (alpha * x).sum(dim=2)
        std         = (alpha * (x - mean.unsqueeze(2)) ** 2).sum(dim=2).clamp(min=1e-9).sqrt()
        return torch.cat([mean, std], dim=1)                   # (B, 2C)


class ECAPAModel(nn.Module):
    """
    Simplified ECAPA-TDNN speaker encoder.

    Reference: Desplanques et al., "ECAPA-TDNN: Emphasized Channel
               Attention, Propagation and Aggregation in TDNN Based
               Speaker Verification", Interspeech 2020.
    """

    def __init__(
        self,
        input_dim:     int = 80,
        channels:      int = 512,
        embedding_dim: int = 192,
        num_speakers:  int = 1211,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=5, padding=2)
        self.input_bn   = nn.BatchNorm1d(channels)

        self.blocks = nn.ModuleList([
            ECAPABlock(channels, kernel_size=3, dilation=2, scale=8),
            ECAPABlock(channels, kernel_size=3, dilation=3, scale=8),
            ECAPABlock(channels, kernel_size=3, dilation=4, scale=8),
        ])

        self.mfa      = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        self.mfa_bn   = nn.BatchNorm1d(channels * 3)
        self.pool     = AttentiveStatsPooling(channels * 3)

        self.fc       = nn.Linear(channels * 6, embedding_dim)
        self.fc_bn    = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def get_embedding(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)                          # (B, D, T)
        x = F.relu(self.input_bn(self.input_proj(x)))

        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)

        x = torch.cat(block_outputs, dim=1)                   # multi-scale fusion
        x = F.relu(self.mfa_bn(self.mfa(x)))
        x = self.pool(x)                                       # (B, channels*6)
        x = self.fc_bn(self.fc(x))
        return x                                               # (B, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_embedding(features))
