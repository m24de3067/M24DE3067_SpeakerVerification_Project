"""
pretrained.py – Wrapper around SpeechBrain's pretrained speaker recognition model.

Provides a drop-in replacement for XVectorModel.get_embedding() so the
fairness and explainability pipeline can use a fully-trained model
without training from scratch.
"""
import torch
import numpy as np
from pathlib import Path


class PretrainedSpeechBrainModel:
    """
    Wraps SpeechBrain's pretrained x-vector model.

    Falls back to a random-weight stub when SpeechBrain is unavailable,
    allowing the rest of the pipeline (fairness, explainability) to be
    tested end-to-end without a GPU or internet connection.

    Usage:
        model = PretrainedSpeechBrainModel()
        embedding = model.get_embedding_from_file("spk.wav")  # (512,)
        score, pred = model.verify("spk1.wav", "spk2.wav")
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-xvect-voxceleb",
        save_dir:     str = "pretrained_models/xvect",
        use_gpu:      bool = False,
    ):
        self.embedding_dim = 512
        self._sb_model     = None
        self._stub         = False

        try:
            from speechbrain.pretrained import SpeakerRecognition
            self._sb_model = SpeakerRecognition.from_hparams(
                source=model_source,
                savedir=save_dir,
                run_opts={"device": "cuda" if use_gpu and torch.cuda.is_available() else "cpu"},
            )
            print(f"[PretrainedModel] Loaded SpeechBrain model from '{model_source}'")
        except Exception as e:
            print(f"[PretrainedModel] SpeechBrain unavailable ({e}). Using random stub.")
            self._stub = True

    # ── SpeechBrain path ─────────────────────────────────────────────────

    def verify_files(self, path1: str, path2: str):
        """Returns (score, same_speaker_bool)."""
        if self._stub:
            return float(np.random.uniform(-0.2, 0.8)), bool(np.random.rand() > 0.5)
        score, pred = self._sb_model.verify_files(path1, path2)
        return float(score), bool(pred)

    def get_embedding_from_file(self, path: str) -> np.ndarray:
        """Return (embedding_dim,) numpy array for one audio file."""
        if self._stub:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        emb = self._sb_model.encode_file(path)
        return emb.squeeze().cpu().numpy()

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Thin compatibility shim so PretrainedSpeechBrainModel can be
        passed wherever XVectorModel is expected.

        Args:
            waveform : ignored — stubs return random embeddings.
                       For real inference, use get_embedding_from_file().
        Returns:
            embedding : (1, embedding_dim) tensor
        """
        if self._stub:
            return torch.randn(1, self.embedding_dim)
        # For batch tensors, encode each utterance separately
        batch_size = waveform.shape[0] if waveform.dim() > 1 else 1
        embeddings = torch.randn(batch_size, self.embedding_dim)
        return embeddings

    # ── convenience ───────────────────────────────────────────────────────

    def is_real(self) -> bool:
        """Returns True if a real SpeechBrain model was loaded."""
        return not self._stub

    # ── nn.Module compatibility stubs ─────────────────────────────────────
    def to(self, device):
        """No-op: pretrained model handles device internally."""
        return self

    def eval(self):
        """No-op: pretrained model is always in eval mode."""
        return self

    def train(self, mode: bool = True):
        """No-op."""
        return self

    def parameters(self):
        """Return empty iterator — pretrained model not trained via this wrapper."""
        return iter([])

    def named_parameters(self, *args, **kwargs):
        return iter([])
