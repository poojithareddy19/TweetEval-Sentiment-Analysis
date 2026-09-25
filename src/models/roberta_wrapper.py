import os
from pathlib import Path

# PyTorch-only wrapper: keep transformers from touching the TensorFlow install
# (Keras 3 breaks its TF integration). Must be set before transformers is imported.
os.environ.setdefault("USE_TF", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.utils.io import check_not_lfs_pointer  # noqa: E402


class RobertaWrapper:

    def __init__(self, model_dir: str = "models/roberta", device: str = None):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {self.model_dir}")

        weights = self.model_dir / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"model.safetensors missing in {self.model_dir}")
        check_not_lfs_pointer(weights, "RoBERTa weights")

        # choose device automatically if not provided
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    def _batchify(self, texts, batch_size=16):
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

    def predict_proba(self, texts, max_length: int = 128, batch_size: int = 16):

        all_probs = []
        for batch in self._batchify(texts, batch_size):
            enc = self.tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = self.model(**enc)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        if not all_probs:
            return np.zeros((0, self.model.config.num_labels))
        return np.vstack(all_probs)
