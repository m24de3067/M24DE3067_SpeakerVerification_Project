"""
fairness.py – Demographic fairness analysis and mitigation strategies.

Implements:
  • Per-group EER computation
  • Fairness gap metric
  • Bootstrap confidence intervals (NIST-style)
  • Threshold calibration (equalized FPR)
  • Data reweighting helper
  • Reporting utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import bootstrap as sp_bootstrap
from sklearn.metrics import roc_curve
from typing import Dict, List, Tuple, Optional

from .evaluate import compute_eer, compute_minDCF


# ══════════════════════════════════════════════════════════════════════════
# Core Fairness Analyzer
# ══════════════════════════════════════════════════════════════════════════

class FairnessAnalyzer:
    """
    Computes and reports fairness metrics across demographic groups.

    Usage:
        fa = FairnessAnalyzer(output_dir="results/fairness")
        results = fa.evaluate(scores, labels, metas)
        fa.plot_comparison(results)
    """

    def __init__(self, output_dir: str = "results/fairness"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── group splitting ───────────────────────────────────────────────────

    def split_by_group(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        metas:  List[Dict],
        group_key: str,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition scores/labels by a metadata key.

        Returns: {group_value: (scores_subset, labels_subset)}
        """
        groups: Dict[str, Tuple[List, List]] = {}
        for score, label, meta in zip(scores, labels, metas):
            gval = str(meta.get(group_key, "unknown"))
            if gval not in groups:
                groups[gval] = ([], [])
            groups[gval][0].append(score)
            groups[gval][1].append(label)
        return {k: (np.array(v[0]), np.array(v[1])) for k, v in groups.items()
                if len(v[0]) > 10}   # drop groups with too few samples

    # ── per-group metrics ─────────────────────────────────────────────────

    def per_group_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        metas:  List[Dict],
        group_keys: List[str] = ("gender1", "nationality1"),
    ) -> Dict:
        """
        Compute EER, minDCF, and 95% CI for each demographic group.

        Returns nested dict:
            {
              "overall": {...},
              "gender1": {
                "male":   {"eer_pct": ..., "ci_low": ..., "ci_high": ..., "n": ...},
                "female": {...},
              },
              ...
              "fairness_gap": {"gender1": ..., "nationality1": ...}
            }
        """
        results = {}

        # Overall
        eer, thresh = compute_eer(scores, labels)
        results["overall"] = {
            "eer_pct":   round(eer * 100, 4),
            "min_dcf":   round(compute_minDCF(scores, labels), 6),
            "threshold": round(thresh, 6),
            "n":         len(labels),
        }

        fairness_gaps = {}

        for key in group_keys:
            groups = self.split_by_group(scores, labels, metas, key)
            results[key] = {}
            group_eers   = []

            for gname, (g_scores, g_labels) in groups.items():
                g_eer, g_thresh = compute_eer(g_scores, g_labels)
                ci_low, ci_high = self._bootstrap_ci(g_scores, g_labels)
                results[key][gname] = {
                    "eer_pct":   round(g_eer * 100, 4),
                    "ci_low":    round(ci_low * 100, 4),
                    "ci_high":   round(ci_high * 100, 4),
                    "threshold": round(g_thresh, 6),
                    "n":         len(g_labels),
                }
                group_eers.append(g_eer)

            if len(group_eers) >= 2:
                fairness_gaps[key] = round((max(group_eers) - min(group_eers)) * 100, 4)

        results["fairness_gap"] = fairness_gaps
        return results

    # ── bootstrap CI ─────────────────────────────────────────────────────

    @staticmethod
    def _bootstrap_ci(
        scores: np.ndarray,
        labels: np.ndarray,
        n_resamples: int = 500,
        confidence:  float = 0.95,
    ) -> Tuple[float, float]:
        """95 % bootstrap confidence interval for EER."""

        def _eer(s, l):
            if len(np.unique(l)) < 2:
                return np.nan
            return compute_eer(s, l)[0]

        try:
            result = sp_bootstrap(
                (scores, labels),
                statistic=lambda s, l: _eer(s, l),
                n_resamples=n_resamples,
                confidence_level=confidence,
                method="percentile",
                paired=True,
                random_state=42,
            )
            return result.confidence_interval.low, result.confidence_interval.high
        except Exception:
            eer = _eer(scores, labels)
            return eer, eer

    # ── threshold calibration ─────────────────────────────────────────────

    def calibrate_thresholds(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        metas:  List[Dict],
        group_key:   str   = "gender1",
        target_fpr:  float = 0.05,
    ) -> Dict[str, float]:
        """
        Equalized-odds threshold calibration.
        Returns per-group decision thresholds that equalise FPR to target_fpr.
        """
        groups = self.split_by_group(scores, labels, metas, group_key)
        thresholds = {}

        for gname, (g_scores, g_labels) in groups.items():
            fpr, tpr, thresh = roc_curve(g_labels, g_scores, pos_label=1)
            idx = np.argmin(np.abs(fpr - target_fpr))
            thresholds[gname] = float(thresh[idx])

        return thresholds

    def apply_calibration(
        self,
        scores:     np.ndarray,
        metas:      List[Dict],
        thresholds: Dict[str, float],
        group_key:  str = "gender1",
    ) -> np.ndarray:
        """Convert scores to binary decisions using per-group thresholds."""
        decisions = np.zeros(len(scores), dtype=int)
        for i, (score, meta) in enumerate(zip(scores, metas)):
            gname    = str(meta.get(group_key, "unknown"))
            thresh   = thresholds.get(gname, thresholds.get("unknown", 0.0))
            decisions[i] = int(score >= thresh)
        return decisions

    # ── reweighting ───────────────────────────────────────────────────────

    @staticmethod
    def compute_sample_weights(
        labels:     List[int],
        groups:     List[str],
    ) -> np.ndarray:
        """
        Inverse-frequency sample weights for balanced group training.
        Returns per-sample weight array summing to len(labels).
        """
        df         = pd.DataFrame({"label": labels, "group": groups})
        group_freq = df["group"].value_counts(normalize=True)
        weights    = df["group"].map(lambda g: 1.0 / group_freq[g]).values
        weights    = weights / weights.sum() * len(weights)
        return weights.astype(np.float32)

    # ── plotting ──────────────────────────────────────────────────────────

    def plot_eer_comparison(
        self,
        before:   Dict,
        after:    Dict,
        group_key: str = "gender1",
        title:     str = "Fairness: EER Before vs After Mitigation",
        filename:  str = "fairness_eer_comparison.png",
    ):
        """Bar chart comparing per-group EER before and after mitigation."""
        if group_key not in before or group_key not in after:
            print(f"[FairnessAnalyzer] No data for group_key='{group_key}'")
            return

        groups  = list(before[group_key].keys())
        eer_b   = [before[group_key][g]["eer_pct"] for g in groups]
        eer_a   = [after[group_key][g]["eer_pct"]  for g in groups]
        ci_b_lo = [before[group_key][g]["ci_low"]  for g in groups]
        ci_b_hi = [before[group_key][g]["ci_high"] for g in groups]
        ci_a_lo = [after[group_key][g]["ci_low"]   for g in groups]
        ci_a_hi = [after[group_key][g]["ci_high"]  for g in groups]

        x     = np.arange(len(groups))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))

        bars_b = ax.bar(x - width / 2, eer_b, width,
                        label="Before mitigation", color="#1B6CA8", alpha=0.85)
        bars_a = ax.bar(x + width / 2, eer_a, width,
                        label="After mitigation",  color="#10B981", alpha=0.85)

        # Error bars (CI)
        err_b = [np.array(eer_b) - np.array(ci_b_lo), np.array(ci_b_hi) - np.array(eer_b)]
        err_a = [np.array(eer_a) - np.array(ci_a_lo), np.array(ci_a_hi) - np.array(eer_a)]
        ax.errorbar(x - width / 2, eer_b, yerr=err_b, fmt="none",
                    color="black", capsize=4, linewidth=1.2)
        ax.errorbar(x + width / 2, eer_a, yerr=err_a, fmt="none",
                    color="black", capsize=4, linewidth=1.2)

        ax.set_xlabel("Demographic group", fontsize=12)
        ax.set_ylabel("EER (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=11)
        ax.legend(fontsize=11)
        ax.axhline(before["overall"]["eer_pct"], color="gray",
                   linestyle="--", linewidth=0.8, label="Overall EER (before)")

        # Annotate fairness gap
        gap_b = before.get("fairness_gap", {}).get(group_key, None)
        gap_a = after.get("fairness_gap",  {}).get(group_key, None)
        if gap_b is not None and gap_a is not None:
            ax.text(0.98, 0.97,
                    f"Fairness gap before: {gap_b:.2f}%\nFairness gap after:  {gap_a:.2f}%",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        plt.tight_layout()
        out = self.output_dir / filename
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[FairnessAnalyzer] Saved plot → {out}")

    def plot_score_distributions(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        metas:  List[Dict],
        group_key: str = "gender1",
        filename:  str = "score_distributions.png",
    ):
        """KDE plots of genuine vs impostor score distributions per group."""
        groups = self.split_by_group(scores, labels, metas, group_key)
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 4), sharey=True)
        if n_groups == 1:
            axes = [axes]

        for ax, (gname, (g_sc, g_lb)) in zip(axes, groups.items()):
            genuine  = g_sc[g_lb == 1]
            impostor = g_sc[g_lb == 0]
            if len(genuine) > 1:
                sns.kdeplot(genuine,  ax=ax, label="Genuine",  color="#1B6CA8", fill=True, alpha=0.4)
            if len(impostor) > 1:
                sns.kdeplot(impostor, ax=ax, label="Impostor", color="#E84545", fill=True, alpha=0.4)
            ax.set_title(f"Group: {gname}", fontweight="bold")
            ax.set_xlabel("Cosine similarity score")
            ax.legend()

        fig.suptitle(f"Score distributions by {group_key}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = self.output_dir / filename
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[FairnessAnalyzer] Saved plot → {out}")

    def save_report(self, results: Dict, filename: str = "fairness_report.csv"):
        """Flatten results dict to CSV."""
        rows = []
        for key, val in results.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    if isinstance(subval, dict):
                        row = {"group_key": key, "group": subkey}
                        row.update(subval)
                        rows.append(row)
                    else:
                        rows.append({"group_key": key, "group": subkey, "value": subval})
        df  = pd.DataFrame(rows)
        out = self.output_dir / filename
        df.to_csv(out, index=False)
        print(f"[FairnessAnalyzer] Saved report → {out}")
        return df
