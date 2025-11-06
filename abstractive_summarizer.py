"""Abstractive summarizer

Provides `abstractive_summarize` which mirrors the extractive API:

- raw_text: input document string
- num_sentences: target number of sentences in the summary
- model_name: seq2seq model to use (HuggingFace model id)
- diversity: kept for API parity (not used heavily in abstractive pipeline)
- top_k_percent: if provided, select target sentences as top proportion of the
  original sentences (rounded up) and aim the abstractive summarizer at that size.

Implementation notes:
- Uses HuggingFace transformers `pipeline("summarization")`.
- Handles long documents by chunking on sentence boundaries, summarizing each
  chunk and then optionally performing a final condensation pass.
- Maps target sentence counts to `min_length`/`max_length` parameters for the
  summarization model by estimating average words per sentence.
"""

from typing import List, Optional
import math

try:
    from transformers import pipeline
except Exception as exc:
    raise ImportError("transformers is required: pip install transformers") from exc

import nltk


def _sent_tokenize(text: str) -> List[str]:
    try:
        return nltk.tokenize.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        return nltk.tokenize.sent_tokenize(text)


def _chunk_sentences(sentences: List[str], max_chars: int) -> List[str]:
    """Group sentences into chunks with at most `max_chars` characters.

    Returns list of chunk strings.
    """
    chunks = []
    cur = []
    cur_len = 0
    for s in sentences:
        slen = len(s)
        if cur and (cur_len + slen + 1 > max_chars):
            chunks.append(" ".join(cur))
            cur = [s]
            cur_len = slen
        else:
            cur.append(s)
            cur_len += slen + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def abstractive_summarize(
    raw_text: str,
    num_sentences: int = 3,
    model_name: str = "facebook/bart-large-cnn",
    diversity: float = 0.7,
    top_k_percent: Optional[float] = None,
    max_chunk_chars: int = 1000,
) -> str:
    """Produce an abstractive summary targeting approximately `num_sentences`.

    top_k_percent behaves the same as in the extractive API: if provided,
    the function computes a target sentence count `k = ceil(n_sentences * pct)`
    and uses that as the target.
    """
    if not raw_text or not raw_text.strip():
        return ""

    sentences = _sent_tokenize(raw_text)
    sentences = [s.strip() for s in sentences if s and len(s.strip()) > 2]
    if not sentences:
        return ""

    n_sent = len(sentences)
    # Determine target sentences k
    if top_k_percent is not None:
        try:
            pct = float(top_k_percent)
        except Exception:
            raise ValueError("top_k_percent must be a number between 0 and 100")
        if not (0 < pct <= 100):
            raise ValueError("top_k_percent must be > 0 and <= 100")
        k = max(1, int(math.ceil(n_sent * (pct / 100.0))))
    else:
        k = int(num_sentences)

    if n_sent <= k:
        # Nothing to summarize, return original
        return " ".join(sentences)

    # estimate average words per sentence to map to token lengths
    total_words = sum(len(s.split()) for s in sentences)
    avg_words_per_sent = max(1.0, total_words / n_sent)

    # heuristic mapping: aim for avg_words_per_sent * k words
    target_words = int(max(10, round(avg_words_per_sent * k)))
    # set min/max for summarizer (tokens ~ words for reasonable estimations)
    min_length = max(5, int(target_words * 0.5))
    max_length = max(min_length + 5, int(target_words * 1.6))

    # build summarization pipeline
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    summarizer = pipeline("summarization", model=model_name, device=device)

    # chunk large documents and summarize each chunk
    chunks = _chunk_sentences(sentences, max_chunk_chars)
    chunk_summaries = []
    for chunk in chunks:
        out = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        # pipeline returns list of dicts with 'summary_text'
        text = out[0]["summary_text"] if isinstance(out, list) and out else str(out)
        chunk_summaries.append(text.strip())

    # If multiple chunk summaries, condense them into final summary
    if len(chunk_summaries) == 1:
        final = chunk_summaries[0]
    else:
        joined = " \n\n ".join(chunk_summaries)
        # run one more summarization pass targeting same k
        out = summarizer(joined, max_length=max_length, min_length=min_length, do_sample=False)
        final = out[0]["summary_text"] if isinstance(out, list) and out else str(out)

    return final.strip()


if __name__ == "__main__":
    demo = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field "
        "as the study of intelligent agents: any device that perceives its environment and takes actions that maximize "
        "its chance of successfully achieving its goals. Colloquially, the term 'AI' is often used to describe machines "
        "that mimic cognitive functions that humans associate with other human minds, such as learning and problem solving. "
        "As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect."
    )

    print("Running abstractive demo (bart-large-cnn)...")
    s = abstractive_summarize(demo, num_sentences=2, model_name="facebook/bart-large-cnn")
    print("\nAbstractive summary:\n", s)
