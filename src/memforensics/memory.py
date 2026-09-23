from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class MemoryWrite:
    session_id: int
    seq: int
    content: str
    id: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'id', f"{self.session_id}-{self.seq}")

class MiniLMEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model.eval()
    def embed(self, text: list[str]) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

class MemoryState:
    def __init__(self, embedder: MiniLMEmbedder):
        self.embedder = embedder
        self.writes: list[MemoryWrite] = []
        self.vectors = np.empty((0, 384))
    def add(self, write: MemoryWrite):
        self.writes.append(write)
        vector = self.embedder.embed([write.content])
        self.vectors = np.vstack([self.vectors, vector])
    def search(self, query: str, top_k: int) -> list[MemoryWrite]:
        k = min(top_k, len(self.writes))
        if k == 0:
            return []
        query_vector = self.embedder.embed([query])[0]
        scores = self.vectors @ query_vector
        idx = np.argsort(-scores, kind='stable')[:k]
        return [self.writes[i] for i in idx]

def build_store(writes: Iterable[MemoryWrite], embedder: MiniLMEmbedder) -> MemoryState:
    store = MemoryState(embedder)
    store.writes = list(writes)
    if store.writes:
        store.vectors = embedder.embed([w.content for w in store.writes])
    return store
