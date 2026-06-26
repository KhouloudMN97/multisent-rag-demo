"""
app.py — MultiSent-RAG interactive demo (Gradio)

Surfaces the three things that make this MultiSent-RAG and not generic RAG:
  1. THE VERDICT   — the sentiment label
  2. THE PATH      — cache hit (⚡ from memory) vs fresh (🔍 retrieved + LLM),
                     with the real speed contrast
  3. THE EVIDENCE  — which sentences it reasoned from, and in which languages,
                     including the cross-lingual call-out

Run locally:  python app.py   ->  opens http://127.0.0.1:7860
"""

import time
import gradio as gr

import os
from core.build_store import build as build_store

# On a fresh deploy the vector store doesn't exist yet — build it from examples.json.
if not os.path.exists("chroma_store"):
    print("No vector store found — building it from data/examples.json...")
    build_store()

from core.reader import MultiSentRAG

# ---------------------------------------------------------------------------
# Load the pipeline ONCE at startup (embedder + Groq + Chroma + cache).
# ---------------------------------------------------------------------------
print("Starting MultiSent-RAG demo — loading pipeline...")
rag = MultiSentRAG()

# Tracks a real fresh-computation time from THIS session, used as an honest
# baseline for the speed contrast shown on cache hits (never a fabricated number).
SESSION = {"fresh_baseline_ms": None}


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------
LANG_INFO = {
    "en": ("\U0001F1EC\U0001F1E7", "English"),
    "fr": ("\U0001F1EB\U0001F1F7", "French"),
    "ar": ("\U0001F1F8\U0001F1E6", "Arabic"),
    "es": ("\U0001F1EA\U0001F1F8", "Spanish"),
    "de": ("\U0001F1E9\U0001F1EA", "German"),
    "hi": ("\U0001F1EE\U0001F1F3", "Hindi"),
    "pt": ("\U0001F1F5\U0001F1F9", "Portuguese"),
    "it": ("\U0001F1EE\U0001F1F9", "Italian"),
    "bg": ("\U0001F1E7\U0001F1EC", "Bulgarian"),
    "fa": ("\U0001F1EE\U0001F1F7", "Persian"),
    "ja": ("\U0001F1EF\U0001F1F5", "Japanese"),
    "zh": ("\U0001F1E8\U0001F1F3", "Chinese"),
    "fa/ar": ("\U0001F310", "Arabic/Persian"),
}


def lang_flag(code):
    return LANG_INFO.get(code, ("\U0001F310", ""))[0]


def lang_name(code):
    return LANG_INFO.get(code, ("\U0001F310", code or "unknown"))[1]


def detect_lang_coarse(text):
    """Dependency-free, script-based guess. Reliable for non-Latin scripts;
    returns None for Latin scripts (ambiguous across en/fr/es/de/pt/it)."""
    def has(lo, hi):
        return any(lo <= ord(c) <= hi for c in text)

    if has(0x3040, 0x30FF):                       # hiragana / katakana
        return "ja"
    if has(0x0600, 0x06FF) or has(0x0750, 0x077F) or has(0xFB50, 0xFDFF):
        return "fa/ar"                            # arabic script (ar or fa)
    if has(0x0400, 0x04FF):                        # cyrillic
        return "bg"
    if has(0x0900, 0x097F):                        # devanagari
        return "hi"
    if has(0x4E00, 0x9FFF):                        # cjk (checked after kana)
        return "zh"
    return None


def _same_family(a, b):
    """Treat the ambiguous 'fa/ar' bucket as the same family as ar or fa."""
    if a == b:
        return True
    fam = {"fa/ar", "fa", "ar"}
    return a in fam and b in fam


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
VERDICT_STYLE = {
    "positive": ("#ecfdf5", "#047857", "#10b981"),
    "negative": ("#fef2f2", "#b91c1c", "#ef4444"),
    "neutral":  ("#f1f5f9", "#475569", "#94a3b8"),
    "uncertain": ("#f5f3ff", "#6d28d9", "#a78bfa"),
}

LABEL_CHIP = {
    "positive": ("#dcfce7", "#166534"),
    "negative": ("#fee2e2", "#991b1b"),
    "neutral":  ("#e2e8f0", "#475569"),
    "uncertain": ("#ede9fe", "#6d28d9"),
}


def _trunc(text, n=90):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def _label_chip(label):
    bg, fg = LABEL_CHIP.get(label, LABEL_CHIP["uncertain"])
    return (
        f'<span style="background:{bg};color:{fg};font-size:12px;'
        f'font-weight:600;padding:2px 8px;border-radius:999px;">{label}</span>'
    )


def _example_row(ex):
    flag = lang_flag(ex.get("language"))
    code = ex.get("language") or ""
    return (
        '<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
        'border-top:1px solid #f1f5f9;">'
        f'<span style="font-size:18px;min-width:24px;">{flag}</span>'
        f'<span style="font-size:11px;color:#94a3b8;min-width:22px;'
        f'text-transform:uppercase;">{code}</span>'
        f'<span style="flex:1;color:#334155;font-size:14px;">{_trunc(ex["text"])}</span>'
        f'{_label_chip(ex["label"])}'
        "</div>"
    )


def _crosslingual_note(result, input_lang):
    if result["source"] == "cache":
        matched = (result.get("cache_hit") or {}).get("language")
        if matched and input_lang and not _same_family(matched, input_lang):
            return (
                f"\U0001F30D <b>Cross-lingual cache hit</b> — your "
                f"{lang_flag(input_lang)} {lang_name(input_lang)} input reused a "
                f"classification first made for a {lang_flag(matched)} "
                f"{lang_name(matched)} sentence, in a shared multilingual space."
            )
        return (
            "\u26A1 <b>Cache hit</b> — reused a previous, semantically similar "
            "classification, skipping both retrieval and the model."
        )

    retrieved = result.get("retrieved") or []
    seen = list(dict.fromkeys(r["language"] for r in retrieved))
    others = [
        l for l in seen
        if input_lang and not _same_family(l, input_lang)
    ]
    if input_lang and others:
        names = ", ".join(f"{lang_flag(l)} {lang_name(l)}" for l in others)
        return (
            f"\U0001F30D <b>Cross-lingual retrieval</b> — classified your "
            f"{lang_flag(input_lang)} {lang_name(input_lang)} input using examples "
            f"in {names}."
        )
    if len(seen) > 1:
        names = ", ".join(f"{lang_flag(l)} {lang_name(l)}" for l in seen)
        return (
            f"\U0001F30D <b>Multilingual retrieval</b> — drew on examples across "
            f"{len(seen)} languages: {names}."
        )
    if seen:
        return f"\U0001F50D Retrieved from {lang_flag(seen[0])} {lang_name(seen[0])} examples."
    return ""


def _speed_block(source, elapsed_ms):
    ms = int(round(elapsed_ms))
    if source == "cache":
        baseline = SESSION["fresh_baseline_ms"]
        badge = (
            '<span style="background:#eef2ff;color:#4338ca;font-weight:600;'
            'font-size:13px;padding:4px 12px;border-radius:999px;">'
            f'\u26A1 From memory \u00B7 {ms} ms</span>'
        )
        if baseline:
            base_ms = int(round(baseline))
            speedup = max(1, round(baseline / max(elapsed_ms, 1)))
            mem_w = max(2, int(100 * elapsed_ms / baseline))
            bars = (
                '<div style="margin-top:10px;font-size:12px;color:#64748b;">'
                '<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                '<span style="min-width:64px;">memory</span>'
                f'<span style="display:inline-block;height:8px;width:{mem_w}px;'
                'background:#6366f1;border-radius:4px;"></span>'
                f'<span>{ms} ms</span></div>'
                '<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                '<span style="min-width:64px;">fresh</span>'
                '<span style="display:inline-block;height:8px;width:100px;'
                'background:#cbd5e1;border-radius:4px;"></span>'
                f'<span>~{base_ms} ms</span></div>'
                f'<div style="margin-top:4px;color:#4338ca;font-weight:600;">'
                f'\u2248 {speedup}\u00D7 faster than computing fresh '
                f'(measured this session)</div>'
                '</div>'
            )
            return badge + bars
        return badge
    return (
        '<span style="background:#f8fafc;color:#475569;font-weight:600;'
        'font-size:13px;padding:4px 12px;border-radius:999px;border:1px solid #e2e8f0;">'
        f'\U0001F50D Computed fresh \u00B7 {ms} ms</span>'
    )


def render_result(result, elapsed_ms, input_lang):
    label = result["label"]
    bg, fg, dot = VERDICT_STYLE.get(label, VERDICT_STYLE["uncertain"])

    verdict = (
        f'<div style="display:inline-flex;align-items:center;gap:10px;'
        f'background:{bg};color:{fg};padding:10px 18px;border-radius:14px;'
        f'font-size:22px;font-weight:700;">'
        f'<span style="height:12px;width:12px;border-radius:50%;background:{dot};"></span>'
        f'{label.upper()}</div>'
    )

    note = _crosslingual_note(result, input_lang)
    note_html = (
        f'<div style="margin:14px 0;padding:12px 14px;background:#f8fafc;'
        f'border-left:3px solid #6366f1;border-radius:8px;color:#334155;'
        f'font-size:14px;line-height:1.5;">{note}</div>'
        if note else ""
    )

    if result["source"] == "cache":
        ev = result.get("cache_hit") or {}
        dist = ev.get("distance")
        ev_title = (
            "Reused this earlier classification"
            + (f' \u00B7 distance {dist}' if dist is not None else "")
        )
        ev_rows = _example_row(ev)
    else:
        ev_title = "Reasoned from these retrieved examples"
        rows = result.get("retrieved") or []
        ev_rows = "".join(_example_row(r) for r in rows) or (
            '<div style="color:#94a3b8;padding:8px 0;">No examples retrieved.</div>'
        )

    evidence = (
        '<div style="margin-top:16px;">'
        f'<div style="font-size:12px;font-weight:600;color:#94a3b8;'
        f'text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px;">'
        f'{ev_title}</div>'
        f'{ev_rows}</div>'
    )

    return (
        '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;'
        'padding:22px 24px;box-shadow:0 1px 3px rgba(0,0,0,.04);">'
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'gap:16px;flex-wrap:wrap;">'
        f'<div>{verdict}</div>'
        f'<div style="text-align:right;">{_speed_block(result["source"], elapsed_ms)}</div>'
        '</div>'
        f'{note_html}'
        f'{evidence}'
        '</div>'
    )


def _empty_state(msg=None):
    msg = msg or "Type a sentence in any of 12 languages and press Analyze."
    return (
        '<div style="background:#fafafa;border:1px dashed #e2e8f0;border-radius:16px;'
        'padding:32px;text-align:center;color:#94a3b8;font-size:14px;">'
        f'{msg}</div>'
    )


# ---------------------------------------------------------------------------
# Core action
# ---------------------------------------------------------------------------
def analyze(text, lang_hint=None):
    text = (text or "").strip()
    if not text:
        return _empty_state("Please enter a sentence first.")

    input_lang = lang_hint or detect_lang_coarse(text)

    t0 = time.perf_counter()
    result = rag.predict(text, language=input_lang)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if result["source"] == "vector_db":
        SESSION["fresh_baseline_ms"] = elapsed_ms  # honest, real fresh time

    return render_result(result, elapsed_ms, input_lang)


def reset_memory():
    rag.cache.embeddings.clear()
    rag.cache.entries.clear()
    SESSION["fresh_baseline_ms"] = None
    return "", _empty_state("Memory cleared. The cache is now empty.")


# ---------------------------------------------------------------------------
# Examples (the cross-lingual cache pair is ① then ②)
# ---------------------------------------------------------------------------
EX_EN = "I absolutely love this, it works perfectly!"
EX_FR = "J'adore \u00e7a, \u00e7a marche parfaitement !"
EX_JA = "\u88fd\u54c1\u306f\u58ca\u308c\u3066\u5c4a\u304d\u307e\u3057\u305f\u3002"
EX_DE = "Der Service war ausgezeichnet und sehr freundlich."


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 900px !important; margin: auto !important;}
#msr-title {font-size: 26px; font-weight: 800;
  color: var(--body-text-color, #0f172a); margin-bottom: 2px;}
#msr-sub {color: var(--body-text-color-subdued, #64748b); font-size: 14px;}
#msr-logo {display:inline-flex;align-items:center;justify-content:center;
  height:40px;width:40px;border-radius:12px;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;font-weight:800;
  font-size:15px;margin-right:10px;vertical-align:middle;}
.msr-hint {color: var(--body-text-color-subdued, #94a3b8);
  font-size:12.5px;margin:2px 0 8px;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS, title="MultiSent-RAG") as demo:
    gr.HTML(
        '<div id="msr-header" style="padding:6px 0 10px;">'
        '<div><span id="msr-logo">MS</span>'
        '<span id="msr-title">MultiSent-RAG</span></div>'
        '<div id="msr-sub">Training-free multilingual sentiment, with a semantic '
        'cache memory and cross-lingual retrieval \u00b7 12 languages</div>'
        '</div>'
    )

    with gr.Row():
        input_box = gr.Textbox(
            label="",
            placeholder="Type a sentence in any of 12 languages\u2026",
            lines=2,
            scale=5,
        )
    with gr.Row():
        analyze_btn = gr.Button("Analyze", variant="primary", scale=3)
        reset_btn = gr.Button("Reset memory", scale=1)

    gr.HTML(
        '<div class="msr-hint">\u2728 To see the cross-lingual cache: click '
        '<b>\u2460</b> first (computed fresh), then <b>\u2461</b> \u2014 the French '
        'sentence reuses the English answer from memory.</div>'
    )
    with gr.Row():
        ex1 = gr.Button("\u2460 \U0001F1EC\U0001F1E7 I love this, works perfectly", size="sm")
        ex2 = gr.Button("\u2461 \U0001F1EB\U0001F1F7 J'adore \u00e7a (same meaning)", size="sm")
        ex3 = gr.Button("\U0001F1EF\U0001F1F5 \u88fd\u54c1\u306f\u58ca\u308c\u3066\u2026", size="sm")
        ex4 = gr.Button("\U0001F1E9\U0001F1EA Der Service war ausgezeichnet", size="sm")

    result = gr.HTML(_empty_state())

    # --- wiring ---
    analyze_btn.click(fn=lambda t: analyze(t, None), inputs=input_box, outputs=result)
    input_box.submit(fn=lambda t: analyze(t, None), inputs=input_box, outputs=result)
    reset_btn.click(fn=reset_memory, outputs=[input_box, result])

    ex1.click(fn=lambda t=EX_EN: (t, analyze(t, "en")), outputs=[input_box, result])
    ex2.click(fn=lambda t=EX_FR: (t, analyze(t, "fr")), outputs=[input_box, result])
    ex3.click(fn=lambda t=EX_JA: (t, analyze(t, "ja")), outputs=[input_box, result])
    ex4.click(fn=lambda t=EX_DE: (t, analyze(t, "de")), outputs=[input_box, result])


if __name__ == "__main__":
    demo.launch()