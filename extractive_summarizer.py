"""Advanced extractive summarizer using sentence-transformers and MMR.

This module accepts a raw text string and returns an extractive summary.
It uses sentence-transformers to compute contextual sentence embeddings
and Maximal Marginal Relevance (MMR) to select diverse, high-coverage
sentences while avoiding redundancy.

API:
- extractive_summarize(raw_text, num_sentences=3, model_name='all-MiniLM-L6-v2', diversity=0.7)

Notes:
- This implementation requires the `sentence-transformers` package.
  Install with `pip install sentence-transformers`.
- The chosen default model (`all-MiniLM-L6-v2`) is compact and fast;
  you may select a larger model for higher-quality embeddings.
"""

from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - helpful install message
    raise ImportError(
        "sentence-transformers is required: pip install sentence-transformers"
    ) from exc

import nltk


def _sent_tokenize(text: str) -> List[str]:
    try:
        return nltk.tokenize.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        return nltk.tokenize.sent_tokenize(text)


_MODEL_CACHE = {}


def _get_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache a SentenceTransformer model by name."""
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def _mmr(doc_embedding: np.ndarray, sent_embeddings: np.ndarray, k: int, diversity: float = 0.7) -> List[int]:
    """Maximal Marginal Relevance selection.

    - doc_embedding: embedding representing the whole document (shape: D)
    - sent_embeddings: embeddings for each sentence (N x D)
    - k: number of sentences to select
    - diversity: lambda in MMR; higher means more relevance, lower means more diversity

    Returns indices of selected sentences.
    """
    # Normalize embeddings to unit vectors for cosine similarity via dot product
    if sent_embeddings.ndim == 1:
        sent_embeddings = sent_embeddings.reshape(1, -1)

    # compute similarities
    sent_embeddings_norm = sent_embeddings / (np.linalg.norm(sent_embeddings, axis=1, keepdims=True) + 1e-8)
    doc_emb_norm = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-8)

    # relevance: similarity between each sentence and the document
    relevance = np.dot(sent_embeddings_norm, doc_emb_norm)

    selected = []
    candidates = list(range(len(sent_embeddings)))

    # pick the highest relevance as first
    first = int(np.argmax(relevance))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        mmr_scores = []
        for c in candidates:
            # relevance component
            rel = relevance[c]
            # redundancy component: max similarity to already selected
            red = max(np.dot(sent_embeddings_norm[c], sent_embeddings_norm[s]) for s in selected)
            score = diversity * rel - (1 - diversity) * red
            mmr_scores.append(score)

        # pick candidate with highest MMR score
        best_idx = int(candidates[np.argmax(mmr_scores)])
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


def extractive_summarize(
    raw_text: str,
    num_sentences: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
    diversity: float = 0.7,
    top_k_percent: float = None,
) -> str:
    """Return an extractive summary using embeddings + MMR.

    Parameters
    - raw_text: input document string
    - num_sentences: number of sentences to return
    - model_name: sentence-transformers model name
    - diversity: MMR diversity parameter in [0,1]; higher => more relevance

    Returns
    - summary string with selected sentences in original order
    """
    if not raw_text or not raw_text.strip():
        return ""

    sentences = _sent_tokenize(raw_text)
    sentences = [s.strip() for s in sentences if s and len(s.strip()) > 2]

    if not sentences:
        return ""

    # Determine number of sentences to select (k)
    n_sent = len(sentences)
    if top_k_percent is not None:
        try:
            top_k_percent = float(top_k_percent)
        except Exception:
            raise ValueError("top_k_percent must be a number between 0 and 100")
        if not (0 < top_k_percent <= 100):
            raise ValueError("top_k_percent must be > 0 and <= 100")
        # compute at least 1 sentence
        k = max(1, int(np.ceil(n_sent * (top_k_percent / 100.0))))
    else:
        k = int(num_sentences)

    # If fewer sentences exist than requested, return all
    if n_sent <= k:
        return " ".join(sentences)

    model = _get_model(model_name)
    sent_embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

    # document representation: mean of sentence embeddings
    doc_embedding = np.mean(sent_embeddings, axis=0)

    # select indices using MMR
    selected_idx = _mmr(doc_embedding, sent_embeddings, k=k, diversity=diversity)

    # preserve original order
    selected_idx_sorted = sorted(selected_idx)
    summary_sentences = [sentences[i] for i in selected_idx_sorted]
    return " ".join(summary_sentences)


if __name__ == "__main__":
    demo = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field "
        "as the study of intelligent agents: any device that perceives its environment and takes actions that maximize "
        "its chance of successfully achieving its goals. Colloquially, the term 'AI' is often used to describe machines "
        "that mimic cognitive functions that humans associate with other human minds, such as learning and problem solving. "
        "As machines become increasingly capable, tasks considered to require "
        "intelligence are often removed from the definition of AI, a phenomenon known as the AI effect."
    )

    print("Demo input:\n", demo)
    out = extractive_summarize(demo, num_sentences=2, model_name="all-MiniLM-L6-v2", diversity=0.6)
    print("\nSummary:\n", out)
