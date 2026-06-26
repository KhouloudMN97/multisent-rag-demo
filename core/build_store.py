"""
build_store.py — loads your owned examples into a Chroma vector store.

Run this ONCE (or whenever you change examples.json).
It reads data/examples.json, embeds each sentence with the multilingual
model, and saves everything into a local Chroma database folder.
The reader will then QUERY this store at question time.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

# Where things live
EXAMPLES_PATH = "data/examples.json"
STORE_DIR = "chroma_store"               # the vector DB will be saved here
COLLECTION_NAME = "sentiment_examples"
EMBEDDER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def build():
    # 1) Load your owned examples
    with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples.")

    # 2) Load the multilingual embedder (CPU, no GPU)
    print("Loading embedder...")
    embedder = SentenceTransformer(EMBEDDER_NAME)

    # 3) Turn each sentence into a vector
    texts = [ex["text"] for ex in examples]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

    # 4) Open a persistent Chroma store (saved to disk in STORE_DIR)
    client = chromadb.PersistentClient(path=STORE_DIR)

    # Start clean: remove an old collection if we're rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # cosine similarity matches our normalized embeddings
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 5) Add everything to the store
    collection.add(
        ids=[str(i) for i in range(len(examples))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"label": ex["label"], "language": ex["language"]}
            for ex in examples
        ],
    )

    print(f"Done. Vector store with {collection.count()} items saved to '{STORE_DIR}/'.")


if __name__ == "__main__":
    build()