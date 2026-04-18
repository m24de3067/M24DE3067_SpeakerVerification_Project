"""
tests/test_pipeline.py – Unit tests for all pipeline components.
Run with: pytest tests/ -v
"""
import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import FeatureExtractor
from src.model import XVectorModel, ECAPAModel
from src.augment import AudioAugmenter, GaussianNoiseAugmenter
from src.evaluate import compute_eer, compute_minDCF, Evaluator
from src.fairness import FairnessAnalyzer
from src.dataset import VoxCelebDataset, VerificationDataset, DemoDataset


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_waveform():
    """2-second 16 kHz dummy audio (silence + noise)."""
    return torch.randn(32000) * 0.01


@pytest.fixture
def extractor():
    return FeatureExtractor(feature_type="fbank", n_mels=80)


@pytest.fixture
def xvector_model():
    m = XVectorModel(input_dim=80, embedding_dim=512, num_speakers=50)
    m.eval()   # eval mode: BatchNorm uses running stats, works with batch_size=1
    return m


# ══════════════════════════════════════════════════════════════════════════
# Feature Extractor Tests
# ══════════════════════════════════════════════════════════════════════════

class TestFeatureExtractor:

    def test_fbank_output_shape(self, dummy_waveform, extractor):
        feat = extractor(dummy_waveform.unsqueeze(0))
        assert feat.dim() == 3, "Expected (batch, time, mel)"
        assert feat.shape[0] == 1
        assert feat.shape[2] == 80

    def test_mfcc_output_shape(self, dummy_waveform):
        mfcc_ext = FeatureExtractor(feature_type="mfcc", n_mfcc=40)
        feat = mfcc_ext(dummy_waveform.unsqueeze(0))
        assert feat.shape[2] == 40

    def test_normalization(self, dummy_waveform, extractor):
        feat = extractor(dummy_waveform.unsqueeze(0))
        mean = feat.mean().item()
        assert abs(mean) < 0.5, "Features should be near zero-mean after CMVN"

    def test_batch_processing(self, extractor):
        batch = torch.randn(4, 32000) * 0.01
        feats = extractor(batch)
        assert feats.shape[0] == 4

    def test_unknown_feature_type(self):
        with pytest.raises(ValueError):
            FeatureExtractor(feature_type="unknown")


# ══════════════════════════════════════════════════════════════════════════
# Model Tests
# ══════════════════════════════════════════════════════════════════════════

class TestXVectorModel:

    def test_embedding_shape(self, dummy_waveform, extractor, xvector_model):
        feat = extractor(dummy_waveform.unsqueeze(0))
        emb  = xvector_model.get_embedding(feat)
        assert emb.shape == (1, 512)

    def test_forward_shape(self, dummy_waveform, extractor, xvector_model):
        feat   = extractor(dummy_waveform.unsqueeze(0))
        logits = xvector_model(feat)
        assert logits.shape == (1, 50)

    def test_embedding_normalizable(self, dummy_waveform, extractor, xvector_model):
        feat = extractor(dummy_waveform.unsqueeze(0))
        emb  = xvector_model.get_embedding(feat)
        norm = torch.nn.functional.normalize(emb, dim=-1)
        assert abs(norm.norm().item() - 1.0) < 1e-4

    def test_gradient_flow(self, dummy_waveform, extractor):
        # Use batch_size=2 for gradient flow test so BatchNorm works in train mode
        model = XVectorModel(input_dim=80, embedding_dim=512, num_speakers=50)
        model.train()
        batch  = torch.stack([dummy_waveform, dummy_waveform * 0.9])
        feat   = extractor(batch)
        logits = model(feat)
        loss   = logits.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_batch_inference(self, extractor):
        model = XVectorModel(input_dim=80, embedding_dim=512, num_speakers=50)
        model.eval()
        batch = torch.randn(4, 32000) * 0.01
        feat  = extractor(batch)
        embs  = model.get_embedding(feat)
        assert embs.shape == (4, 512)


class TestECAPAModel:

    def test_embedding_shape(self, dummy_waveform, extractor):
        ecapa = ECAPAModel(input_dim=80, embedding_dim=192, num_speakers=50)
        ecapa.eval()
        feat  = extractor(dummy_waveform.unsqueeze(0))
        emb   = ecapa.get_embedding(feat)
        assert emb.shape == (1, 192)

    def test_forward_output(self, dummy_waveform, extractor):
        ecapa  = ECAPAModel(input_dim=80, num_speakers=50)
        ecapa.eval()
        feat   = extractor(dummy_waveform.unsqueeze(0))
        logits = ecapa(feat)
        assert logits.shape[1] == 50


# ══════════════════════════════════════════════════════════════════════════
# Augmentation Tests
# ══════════════════════════════════════════════════════════════════════════

class TestAugmentation:

    def test_gaussian_augmenter(self, dummy_waveform):
        aug    = GaussianNoiseAugmenter(noise_std=0.005)
        noisy  = aug(dummy_waveform)
        assert noisy.shape == dummy_waveform.shape
        assert not torch.allclose(noisy, dummy_waveform)

    def test_audio_augmenter_no_corpus(self, dummy_waveform):
        aug    = AudioAugmenter(musan_path=None, rir_path=None, noise_prob=1.0)
        result = aug(dummy_waveform)
        assert result.shape == dummy_waveform.shape

    def test_augmenter_batch(self):
        aug   = GaussianNoiseAugmenter()
        batch = torch.randn(4, 32000) * 0.01
        out   = torch.stack([aug(w) for w in batch])
        assert out.shape == batch.shape


# ══════════════════════════════════════════════════════════════════════════
# Evaluation Tests
# ══════════════════════════════════════════════════════════════════════════

class TestEvaluation:

    def test_compute_eer_balanced(self):
        np.random.seed(42)
        genuine  = np.random.normal(0.7, 0.1, 200)
        impostor = np.random.normal(0.3, 0.1, 200)
        scores   = np.concatenate([genuine, impostor])
        labels   = np.array([1]*200 + [0]*200)
        eer, thresh = compute_eer(scores, labels)
        assert 0.0 <= eer <= 0.5, "EER should be between 0 and 0.5"
        assert 0.3 <= thresh <= 0.7

    def test_compute_eer_perfect(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1,   1,   0,   0  ])
        eer, _ = compute_eer(scores, labels)
        assert eer < 0.05, "Perfect separation should give near-zero EER"

    def test_compute_minDCF(self):
        scores = np.random.randn(200)
        labels = (scores > 0).astype(int)
        dcf    = compute_minDCF(scores, labels)
        assert 0.0 <= dcf <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# Fairness Tests
# ══════════════════════════════════════════════════════════════════════════

class TestFairness:

    def _make_data(self, n=200):
        np.random.seed(0)
        scores = np.random.randn(n)
        labels = (scores > 0).astype(int)
        metas  = [
            {"gender1": "male" if i % 2 == 0 else "female",
             "nationality1": "native" if i % 3 == 0 else "non_native"}
            for i in range(n)
        ]
        return scores, labels, metas

    def test_per_group_metrics(self, tmp_path):
        fa = FairnessAnalyzer(output_dir=str(tmp_path))
        scores, labels, metas = self._make_data()
        results = fa.per_group_metrics(scores, labels, metas, group_keys=["gender1"])
        assert "overall" in results
        assert "gender1" in results
        assert "male" in results["gender1"] or "female" in results["gender1"]

    def test_calibrate_thresholds(self, tmp_path):
        fa = FairnessAnalyzer(output_dir=str(tmp_path))
        scores, labels, metas = self._make_data()
        thresholds = fa.calibrate_thresholds(scores, labels, metas, group_key="gender1")
        assert "male" in thresholds or "female" in thresholds

    def test_fairness_gap_is_nonnegative(self, tmp_path):
        fa = FairnessAnalyzer(output_dir=str(tmp_path))
        scores, labels, metas = self._make_data()
        results = fa.per_group_metrics(scores, labels, metas, group_keys=["gender1"])
        gap = results.get("fairness_gap", {}).get("gender1", 0)
        assert gap >= 0.0

    def test_sample_weights(self):
        labels = [1, 0, 1, 0, 1]
        groups = ["male", "female", "male", "female", "male"]
        weights = FairnessAnalyzer.compute_sample_weights(labels, groups)
        assert weights.shape[0] == 5
        assert weights.sum() > 0


# ══════════════════════════════════════════════════════════════════════════
# Dataset Tests
# ══════════════════════════════════════════════════════════════════════════

class TestDatasets:

    def test_voxceleb_synthetic(self):
        ds = VoxCelebDataset(csv_path="nonexistent.csv", synthetic=True)
        assert len(ds) > 0
        wav, label = ds[0]
        assert wav.shape[0] == VoxCelebDataset.FIXED_LEN
        assert isinstance(label, int)

    def test_verification_synthetic(self):
        ds = VerificationDataset(csv_path="nonexistent.csv", synthetic=True)
        assert len(ds) > 0
        wav1, wav2, label, meta = ds[0]
        assert wav1.shape == wav2.shape
        assert label in (0, 1)
        assert "gender1" in meta

    def test_demo_split(self):
        ds     = VerificationDataset(csv_path="nonexistent.csv", synthetic=True)
        splits = DemoDataset.split(ds)
        assert isinstance(splits, dict)
        for group in ["male", "female", "native", "non_native"]:
            assert group in splits

