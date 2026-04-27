from __future__ import annotations

from sentence_transformers import SentenceTransformer
import heapq
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)


def compute_text_pairs_and_topk(
    resume_df: pd.DataFrame,
    job_df: pd.DataFrame,
    resume_text_col: str = "resume_cleaned",
    job_desc_col: str = "job_summary_cleaned",
    top_k: int = 5,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute semantic/text similarity between resumes and jobs.

    Supported model_name values:
    - "tfidf"
    - any SentenceTransformer model name, e.g.
      - "BAAI/bge-small-en-v1.5"
      - "all-MiniLM-L6-v2"
      - "all-mpnet-base-v2"

    Returns:
        pair_df: all resume-job pair scores
        topk_df: top-k jobs per resume
    """

    if resume_text_col not in resume_df.columns:
        raise KeyError(f"Column '{resume_text_col}' not found in resume_df")
    if job_desc_col not in job_df.columns:
        raise KeyError(f"Column '{job_desc_col}' not found in job_df")

    job_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []

    # Prepare job texts
    job_texts = [str(jr.get(job_desc_col, "")) for _, jr in job_df.iterrows()]

    # ---------- TF-IDF branch ----------
    if model_name.lower() == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        job_matrix = vectorizer.fit_transform(job_texts)

        for idx, (_, jr) in enumerate(job_df.iterrows()):
            job_rows.append({
                "job_idx": int(jr.name),
                "job_title": jr.get("job_title", ""),
                "company": jr.get("company", ""),
                "job_location": jr.get("job_location", ""),
            })

        for _, rr in resume_df.iterrows():
            resume_id = rr.get("ID", rr.name)
            resume_text = str(rr.get(resume_text_col, ""))

            resume_vector = vectorizer.transform([resume_text])

            scored_rows: List[Dict[str, Any]] = []
            similarities = cosine_similarity(resume_vector, job_matrix)[0]

            for idx, sim in enumerate(similarities):
                j = job_rows[idx]

                row = {
                    "resume_id": resume_id,
                    "resume_idx": int(rr.name),
                    "job_idx": j["job_idx"],
                    "job_title": j["job_title"],
                    "company": j["company"],
                    "job_location": j["job_location"],
                    "model_name": model_name,
                    "semantic_similarity": round(float(sim), 4),
                    "final_score": round(float(sim) * 100, 2),
                }

                pair_rows.append(row)
                scored_rows.append(row)

            best = heapq.nlargest(top_k, scored_rows, key=lambda x: x["final_score"])
            for rank, entry in enumerate(best, start=1):
                out = entry.copy()
                out["rank"] = rank
                topk_rows.append(out)

        return pd.DataFrame(pair_rows), pd.DataFrame(topk_rows)

    # ---------- SentenceTransformer branch ----------
    embedder = TextEmbedder(model_name=model_name)

    # For BGE models, it is often better to add query/passage prefixes
    if "bge" in model_name.lower():
        encoded_job_texts = [f"passage: {text}" for text in job_texts]
    else:
        encoded_job_texts = job_texts

    job_embs = embedder.encode(encoded_job_texts)

    for idx, (_, jr) in enumerate(job_df.iterrows()):
        job_rows.append({
            "job_idx": int(jr.name),
            "job_title": jr.get("job_title", ""),
            "company": jr.get("company", ""),
            "job_location": jr.get("job_location", ""),
            "job_emb": job_embs[idx],
        })

    for _, rr in resume_df.iterrows():
        resume_id = rr.get("ID", rr.name)
        resume_text = str(rr.get(resume_text_col, ""))

        if "bge" in model_name.lower():
            resume_input = f"query: {resume_text}"
        else:
            resume_input = resume_text

        resume_emb = embedder.encode(resume_input)[0]

        scored_rows: List[Dict[str, Any]] = []

        for j in job_rows:
            # embeddings are normalized, so dot product == cosine similarity
            semantic_sim = float(np.dot(resume_emb, j["job_emb"]))

            row = {
                "resume_id": resume_id,
                "resume_idx": int(rr.name),
                "job_idx": j["job_idx"],
                "job_title": j["job_title"],
                "company": j["company"],
                "job_location": j["job_location"],
                "model_name": model_name,
                "semantic_similarity": round(semantic_sim, 4),
                "final_score": round(semantic_sim * 100, 2),
            }

            pair_rows.append(row)
            scored_rows.append(row)

        best = heapq.nlargest(top_k, scored_rows, key=lambda x: x["final_score"])
        for rank, entry in enumerate(best, start=1):
            out = entry.copy()
            out["rank"] = rank
            topk_rows.append(out)

    return pd.DataFrame(pair_rows), pd.DataFrame(topk_rows)