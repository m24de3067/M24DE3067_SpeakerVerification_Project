"""
explain.py – Explainability via Integrated Gradients (Captum).
Visualises which Mel-filterbank bins drive speaker verification decisions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Tuple


class SaliencyExplainer:
    """
    Gradient-based feature attribution for speaker verification models.

    Uses Integrated Gradients to attribute the similarity score of a
    test pair back to input features, revealing which frequency bands
    and time frames most influence the verification decision.

    Args:
        model             : XVectorModel or ECAPAModel instance
        feature_extractor : FeatureExtractor instance
        device            : torch.device
        output_dir        : where to save saliency plots
        n_steps           : number of IG integration steps
    """

    def __init__(
        self,
        model,
        feature_extractor,
        device:     torch.device = None,
        output_dir: str = "results/explainability",
        n_steps:    int = 50,
    ):
        self.model     = model
        self.extractor = feature_extractor
        self.device    = device or torch.device("cpu")
        self.out_dir   = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n_steps   = n_steps
        self.model.to(self.device).eval()

    # ── IG implementation (no Captum dependency needed for demo) ──────────

    def _integrated_gradients(
        self,
        features:   torch.Tensor,
        target_fn,
        baseline:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients manually (Sundararajan et al., 2017).

        Args:
            features  : (1, T, D) input features (requires_grad must be settable)
            target_fn : function mapping features → scalar score
            baseline  : (1, T, D) reference input; defaults to zeros (silence)
        Returns:
            attributions : (1, T, D) attribution tensor
        """
        if baseline is None:
            baseline = torch.zeros_like(features)

        scaled_inputs = [
            baseline + (float(i) / self.n_steps) * (features - baseline)
            for i in range(1, self.n_steps + 1)
        ]

        grads = []
        for x in scaled_inputs:
            x = x.detach().requires_grad_(True)
            score = target_fn(x)
            score.backward()
            grads.append(x.grad.detach().clone())

        avg_grads    = torch.stack(grads).mean(dim=0)
        attributions = (features - baseline) * avg_grads
        return attributions

    # ── public API ────────────────────────────────────────────────────────

    def explain_pair(
        self,
        wav1:       torch.Tensor,
        wav2:       torch.Tensor,
        label:      int = 1,
        utterance_id: str = "pair",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute saliency maps for a verification trial.

        Returns:
            attr1 : (T, D) attribution for utterance 1
            attr2 : (T, D) attribution for utterance 2
        """
        def _embed(wav: torch.Tensor) -> torch.Tensor:
            feat = self.extractor(wav.unsqueeze(0).to(self.device))
            return torch.nn.functional.normalize(
                self.model.get_embedding(feat), dim=-1
            )

        # Target: cosine similarity between the two embeddings
        def _score_fn1(feat1: torch.Tensor) -> torch.Tensor:
            e1 = torch.nn.functional.normalize(self.model.get_embedding(feat1), dim=-1)
            e2 = _embed(wav2).detach()
            return (e1 * e2).sum()

        def _score_fn2(feat2: torch.Tensor) -> torch.Tensor:
            e1 = _embed(wav1).detach()
            e2 = torch.nn.functional.normalize(self.model.get_embedding(feat2), dim=-1)
            return (e1 * e2).sum()

        feat1 = self.extractor(wav1.unsqueeze(0).to(self.device))
        feat2 = self.extractor(wav2.unsqueeze(0).to(self.device))

        attr1 = self._integrated_gradients(feat1.detach(), _score_fn1)
        attr2 = self._integrated_gradients(feat2.detach(), _score_fn2)

        a1 = attr1.squeeze(0).cpu().numpy()  # (T, D)
        a2 = attr2.squeeze(0).cpu().numpy()

        self.plot_saliency_pair(feat1, feat2, a1, a2, label=label, uid=utterance_id)
        return a1, a2

    def explain_single(
        self,
        waveform: torch.Tensor,
        speaker_class: int,
        utterance_id: str = "utt",
    ) -> np.ndarray:
        """
        Compute saliency for speaker classification (single utterance).
        """
        def _score_fn(feat: torch.Tensor) -> torch.Tensor:
            logits = self.model(feat)
            return logits[0, speaker_class]

        feat = self.extractor(waveform.unsqueeze(0).to(self.device))
        attr = self._integrated_gradients(feat.detach(), _score_fn)
        a    = attr.squeeze(0).cpu().numpy()

        self.plot_single_saliency(feat.squeeze(0).cpu().numpy(), a, uid=utterance_id)
        return a

    # ── plotting ──────────────────────────────────────────────────────────

    def plot_saliency_pair(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        attr1: np.ndarray,
        attr2: np.ndarray,
        label: int = 1,
        uid:   str = "pair",
    ):
        f1 = feat1.squeeze(0).cpu().numpy()   # (T, D)
        f2 = feat2.squeeze(0).cpu().numpy()

        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

        verdict = "Same speaker ✓" if label == 1 else "Different speakers ✗"

        for col, (feat, attr, name) in enumerate([
            (f1, attr1, "Utterance 1"),
            (f2, attr2, "Utterance 2"),
        ]):
            # Feature spectrogram
            ax_feat = fig.add_subplot(gs[0, col])
            im = ax_feat.imshow(feat.T, aspect="auto", origin="lower", cmap="viridis")
            ax_feat.set_title(f"{name} — Log-Mel Filterbank", fontsize=10)
            ax_feat.set_xlabel("Frame")
            ax_feat.set_ylabel("Mel bin")
            plt.colorbar(im, ax=ax_feat, shrink=0.8)

            # Attribution heatmap
            ax_attr = fig.add_subplot(gs[1, col])
            abs_attr = np.abs(attr).T
            im2 = ax_attr.imshow(abs_attr, aspect="auto", origin="lower",
                                 cmap="hot", vmin=0)
            ax_attr.set_title(f"{name} — IG Attribution", fontsize=10)
            ax_attr.set_xlabel("Frame")
            ax_attr.set_ylabel("Mel bin")
            plt.colorbar(im2, ax=ax_attr, shrink=0.8)

        fig.suptitle(
            f"Integrated Gradients Saliency  |  {verdict}",
            fontsize=12, fontweight="bold"
        )
        out = self.out_dir / f"saliency_{uid}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SaliencyExplainer] Saved → {out}")

    def plot_single_saliency(
        self,
        feat: np.ndarray,
        attr: np.ndarray,
        uid:  str = "utt",
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

        # Spectrogram
        im1 = ax1.imshow(feat.T, aspect="auto", origin="lower", cmap="viridis")
        ax1.set_title("Log-Mel Filterbank Features")
        ax1.set_ylabel("Mel bin")
        plt.colorbar(im1, ax=ax1)

        # Attribution
        im2 = ax2.imshow(np.abs(attr).T, aspect="auto", origin="lower", cmap="hot")
        ax2.set_title("Integrated Gradients Attribution (|attribution|)")
        ax2.set_ylabel("Mel bin")
        plt.colorbar(im2, ax=ax2)

        # Per-frequency attribution (averaged over time)
        freq_attr = np.abs(attr).mean(axis=0)
        ax3.bar(np.arange(len(freq_attr)), freq_attr, color="#1B6CA8", alpha=0.8)
        ax3.set_title("Average attribution per Mel bin (frequency importance)")
        ax3.set_xlabel("Mel bin")
        ax3.set_ylabel("|Attribution|")

        plt.tight_layout()
        out = self.out_dir / f"saliency_{uid}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SaliencyExplainer] Saved → {out}")

    def frequency_importance_report(
        self,
        attributions: np.ndarray,
        n_mels:       int = 80,
        filename:     str = "frequency_importance.png",
    ):
        """
        Summarise which Mel frequency bands matter most across all samples.
        attributions : (N, T, D) array of per-sample attributions.
        """
        mean_attr = np.abs(attributions).mean(axis=(0, 1))  # (D,)
        hz_approx = np.linspace(0, 8000, n_mels)            # rough Hz mapping

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(hz_approx, mean_attr, alpha=0.7, color="#1B6CA8")
        ax.plot(hz_approx, mean_attr, color="#0D1B2A", linewidth=1.5)
        ax.axvline(125,  color="green",  linestyle="--", alpha=0.6, label="Low freq (125 Hz)")
        ax.axvline(1000, color="orange", linestyle="--", alpha=0.6, label="F0 region (~1 kHz)")
        ax.axvline(3500, color="red",    linestyle="--", alpha=0.6, label="High freq (3.5 kHz)")
        ax.set_xlabel("Approximate frequency (Hz)")
        ax.set_ylabel("Mean |attribution|")
        ax.set_title("Frequency importance for speaker verification\n(regions with high attribution drive decisions)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = self.out_dir / filename
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[SaliencyExplainer] Saved → {out}")
