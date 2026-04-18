"""
evaluate.py – EER computation, PLDA scoring, and verification evaluation.
"""
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════
# EER & Threshold
# ══════════════════════════════════════════════════════════════════════════

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding decision threshold.

    Returns:
        eer       : fraction (multiply by 100 for percentage)
        threshold : score at EER operating point
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find the index where |FPR - FNR| is minimised
    idx       = np.nanargmin(np.abs(fpr - fnr))
    eer       = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def compute_minDCF(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_miss:   float = 1.0,
    c_fa:     float = 1.0,
) -> float:
    """NIST SRE minimum Detection Cost Function."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr  = 1.0 - tpr
    dcf  = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    norm = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(dcf) / norm)


# ══════════════════════════════════════════════════════════════════════════
# Scoring Backends
# ══════════════════════════════════════════════════════════════════════════

def cosine_score(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Cosine similarity between two embeddings."""
    e1 = F.normalize(emb1.float(), dim=-1)
    e2 = F.normalize(emb2.float(), dim=-1)
    return float((e1 * e2).sum())


class PLDAScorer:
    """
    Simplified PLDA (Probabilistic LDA) scoring using sklearn's LDA.
    In production, use a proper PLDA implementation; this is a
    dimensionality-reduction + cosine-score approximation suitable
    for demonstration and fairness analysis.
    """

    def __init__(self, n_components: int = 200):
        self.n_components = n_components
        self.lda          = LinearDiscriminantAnalysis(n_components=n_components)
        self.fitted       = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        self.lda.fit(embeddings, labels)
        self.fitted = True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PLDAScorer must be fitted before scoring.")
        return self.lda.transform(embeddings)

    def score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        t1 = self.transform(emb1[np.newaxis])
        t2 = self.transform(emb2[np.newaxis])
        num = (t1 * t2).sum()
        den = (np.linalg.norm(t1) * np.linalg.norm(t2)) + 1e-9
        return float(num / den)


# ══════════════════════════════════════════════════════════════════════════
# Main Evaluator
# ══════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Runs verification trials, collects scores, and computes metrics.

    Supports:
      • cosine scoring (default)
      • PLDA scoring (optional, requires fitted PLDAScorer)
    """

    def __init__(
        self,
        model,
        feature_extractor,
        device:       torch.device = None,
        plda_scorer:  Optional[PLDAScorer] = None,
        use_pretrained: bool = False,
    ):
        self.model     = model
        self.extractor = feature_extractor
        self.device    = device or torch.device("cpu")
        self.plda      = plda_scorer
        self.use_pretrained = use_pretrained
        self.model.to(self.device).eval()

    @torch.no_grad()
    def get_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        waveform = waveform.unsqueeze(0).to(self.device)
        features = self.extractor(waveform)
        emb      = self.model.get_embedding(features)
        return emb.squeeze(0).cpu().numpy()

    def run_trials(
        self,
        dataloader,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Args:
            dataloader : yields (wav1, wav2, label, meta)
        Returns:
            scores  : (N,) similarity scores
            labels  : (N,) binary labels
            metas   : list of metadata dicts
        """
        all_scores: List[float] = []
        all_labels: List[int]   = []
        all_metas:  List[Dict]  = []

        iterator = tqdm(dataloader, desc="Running trials") if verbose else dataloader

        for wav1_batch, wav2_batch, label_batch, meta_batch in iterator:
            for i in range(wav1_batch.shape[0]):
                e1 = self.get_embedding(wav1_batch[i])
                e2 = self.get_embedding(wav2_batch[i])

                if self.plda and self.plda.fitted:
                    score = self.plda.score(e1, e2)
                else:
                    score = float(
                        np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
                    )

                all_scores.append(score)
                all_labels.append(int(label_batch[i]))
                all_metas.append({k: v[i] if hasattr(v, '__getitem__') else v
                                  for k, v in meta_batch.items()})

        return np.array(all_scores), np.array(all_labels), all_metas

    def full_report(
        self,
        scores:  np.ndarray,
        labels:  np.ndarray,
    ) -> Dict[str, float]:
        eer, thresh = compute_eer(scores, labels)
        minDCF      = compute_minDCF(scores, labels)
        return {
            "eer_pct":    round(eer * 100, 4),
            "min_dcf":    round(minDCF, 6),
            "threshold":  round(thresh, 6),
            "n_trials":   len(labels),
            "n_positive": int(labels.sum()),
            "n_negative": int((1 - labels).sum()),
        }
