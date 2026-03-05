from __future__ import annotations

import hashlib
import math
import os
from typing import List, Optional


class Embedder:
    """Dependency-light embedding backend for local prototyping."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 1) & 1) else -1.0
            vec[idx] += sign
        return self._l2_normalize(vec)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        cur: List[str] = []
        tok: List[str] = []
        for ch in text:
            if ch.isalnum():
                tok.append(ch)
            elif tok:
                cur.append("".join(tok))
                tok = []
        if tok:
            cur.append("".join(tok))
        return cur

    def _l2_normalize(self, v: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class OpenAIEmbedder:
    """OpenAI-backed embedder using text-embedding-3-small (dim=1536)."""

    DIM = 1536

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.dim = self.DIM

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
