---
title: MultiSent-RAG
emoji: 🌍
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# MultiSent-RAG — Interactive Demo

Training-free multilingual sentiment analysis with a **semantic cache memory**
and **cross-lingual retrieval**, across 12 languages.

This demo runs the method from the paper *MultiSent-RAG: A Retrieval and
Memory-Augmented System for Multilingual Sentiment Processing* (Mnassri,
Farahbakhsh, Crespi).

- 📄 Research code: https://github.com/KhouloudMN97/MultiSent-RAG
- ⚡ The semantic cache reuses prior inferences across languages, skipping
  retrieval and the LLM on a hit.
- 🌍 Cross-lingual transfer: a query in one language is classified using
  examples retrieved from others, in a shared multilingual embedding space.

*Note: the LLM is served via the Groq API (instead of a local quantized model),
so the demo runs on CPU with no GPU required.*