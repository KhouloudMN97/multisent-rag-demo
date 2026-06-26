"""
reader.py — MultiSent-RAG inference pipeline (demo version)

Faithful to the paper's Figure 3 flow:
  1. CACHE first: if a semantically-close entry exists -> reuse its label,
     bypassing retrieval AND the LLM.            (the efficiency contribution)
  2. On a MISS: retrieve top-k from the Chroma vector store (semantic search),
     build the few-shot prompt, ask the LLM (Groq) for the sentiment.
  3. STORE the new prediction so similar inputs hit the cache next time.

Infra swaps vs. the paper (method unchanged):
  - LLM served via Groq API instead of a local 4-bit Mistral (no GPU needed)
  - same Chroma vector store, same multilingual embedder
"""

import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from core.semantic_cache import SemanticCache

load_dotenv()

STORE_DIR = "chroma_store"
COLLECTION_NAME = "sentiment_examples"
EMBEDDER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class MultiSentRAG:
    def __init__(
        self,
        model_name="llama-3.3-70b-versatile",
        top_k=7,                 # paper uses k=7
        cache_threshold=0.9,     # paper's angular-distance threshold
    ):
        self.model_name = model_name
        self.top_k = top_k

        # --- shared multilingual embedder (used by retrieval AND the cache) ---
        print("Loading embedder (cached after first run)...")
        self.embedder = SentenceTransformer(EMBEDDER_NAME)

        # --- LLM (generation) ---
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # --- Chroma vector store (retrieval) ---
        chroma_client = chromadb.PersistentClient(path=STORE_DIR)
        self.collection = chroma_client.get_collection(COLLECTION_NAME)

        # --- semantic cache (memory), sharing the same embedder ---
        self.cache = SemanticCache(self.embedder, threshold=cache_threshold)

    # ---------- retrieval ----------
    def retrieve(self, query):
        """Semantic search over the Chroma store. Returns the top-k examples."""
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        res = self.collection.query(
            query_embeddings=query_vec,
            n_results=self.top_k,
        )

        # Repackage Chroma's result into a simple list of dicts
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [
            {"text": d, "label": m["label"], "language": m["language"]}
            for d, m in zip(docs, metas)
        ]

    # ---------- prompt (adapted from paper Appendix A.2, extended to 3 labels) ----------
    def build_prompt(self, query, retrieved):
        examples_block = "\n".join(
            f'Text: "{ex["text"]}"\nSentiment: {ex["label"]}'
            for ex in retrieved
        )
        return f"""You are a multilingual sentiment analysis expert.
Classify the sentiment of the text as exactly one word:
- positive: happiness, satisfaction, praise, optimism
- negative: dissatisfaction, sadness, anger, frustration
- neutral: factual or non-emotional statements

Here are some retrieved examples:

{examples_block}

Now classify this text. Respond with one word only.
Text: "{query}"
Sentiment:"""

    @staticmethod
    def _parse_label(raw):
        raw = raw.strip().lower()
        if "positive" in raw:
            return "positive"
        if "negative" in raw:
            return "negative"
        if "neutral" in raw:
            return "neutral"
        return "uncertain"

    # ---------- full pipeline ----------
    def predict(self, query, language=None):
        # 1) CACHE FIRST
        hit = self.cache.lookup(query)
        if hit is not None:
            return {
                "label": hit["label"],
                "source": "cache",
                "cache_hit": hit,        # includes the matched text + its language
                "retrieved": None,
            }

        # 2) MISS -> retrieve + generate
        retrieved = self.retrieve(query)
        prompt = self.build_prompt(query, retrieved)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        label = self._parse_label(response.choices[0].message.content)

        # 3) STORE for future reuse
        self.cache.store(query, label, language=language)

        return {
            "label": label,
            "source": "vector_db",
            "cache_hit": None,
            "retrieved": retrieved,      # so the UI can show cross-lingual retrieval
        }