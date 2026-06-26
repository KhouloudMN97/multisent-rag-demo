"""
semantic_cache.py — MultiSent-RAG Semantic Cache (Memory Layer)

Faithful to the paper (MultiSent-RAG-Cache):
  - sits IN FRONT of retrieval
  - distance metric = ANGULAR DISTANCE (Eq. 1), range [0, 2], 0 = identical
  - if distance <= threshold -> return cached label, bypassing BOTH
                                Chroma retrieval AND the LLM   (source = "cache")
  - otherwise -> caller runs retrieval + LLM, then store() the result
  - cross-lingual reuse: a query in one language can hit an entry stored for
    another language, because matching is on the multilingual embeddings.

Threshold = 0.9 (the paper's balance point, Appendix B).

Bug fix vs. research code: the original used an Annoy index, which is IMMUTABLE
after .build(), so storing items at runtime crashed. Here embeddings live in a
list and angular distance is computed directly, so the cache can grow live.
Same metric, no immutability trap.
"""

import numpy as np


class SemanticCache:
    def __init__(self, encoder, threshold=0.9):
        self.encoder = encoder
        self.threshold = threshold     # angular-distance threshold (paper: 0.9)
        self.embeddings = []           # L2-normalized query vectors
        self.entries = []              # parallel: {"label", "text", "language"}

    def _embed(self, text):
        # normalized -> dot product equals cosine similarity
        return self.encoder.encode([text], normalize_embeddings=True)[0]

    @staticmethod
    def _angular_distance(cosine):
        # paper Eq. (1): sqrt(2 * (1 - cosine)). clip guards float overshoot.
        return float(np.sqrt(2.0 * (1.0 - np.clip(cosine, -1.0, 1.0))))

    def lookup(self, query):
        """Return the closest cached entry if within threshold, else None."""
        if not self.embeddings:
            return None

        q = self._embed(query)
        cosines = np.array(self.embeddings) @ q
        best = int(np.argmax(cosines))               # closest = highest cosine
        distance = self._angular_distance(cosines[best])

        if distance <= self.threshold:
            hit = dict(self.entries[best])
            hit["distance"] = round(distance, 3)
            return hit                                # {label, text, language, distance}
        return None

    def store(self, query, label, language=None):
        """Save a fresh prediction so similar inputs hit next time."""
        self.embeddings.append(self._embed(query))
        self.entries.append({"label": label, "text": query, "language": language})