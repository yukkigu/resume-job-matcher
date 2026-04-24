from __future__ import annotations

from sentence_transformers import SentenceTransformer
import ast
import heapq
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

ROLE_SYNONYMS: Dict[str, List[str]] = {
    "accountant": [
        "accountant",
        "accounting",
        "bookkeeper",
        "finance",
        "financial analyst",
        "staff accountant",
        "senior accountant",
        "accounts payable",
        "accounts receivable",
    ],
    "data science": [
        "data scientist",
        "data analyst",
        "machine learning engineer",
        "ml engineer",
        "ai engineer",
        "analytics",
        "business intelligence",
    ],
    "hr": [
        "human resources",
        "hr",
        "recruiter",
        "talent acquisition",
        "people operations",
        "hr specialist",
    ],
    "software engineer": [
        "software engineer",
        "software developer",
        "developer",
        "programmer",
        "backend engineer",
        "frontend engineer",
        "full stack",
    ],
    "business analyst": [
        "business analyst",
        "analyst",
        "operations analyst",
        "reporting analyst",
    ],
}

class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)


def compute_text_pairs_and_topk(
    resume_df: pd.DataFrame,
    job_df: pd.DataFrame,
    resume_text_col: str = "resume_cleaned",
    job_desc_col: str = "job_summary",
    top_k: int = 5,
    model_name: str = "BAAI/bge-small-en-v1.5",
):
    embedder = TextEmbedder(model_name=model_name)

    job_rows: List[Dict[str, Any]] = []

    job_texts = []
    for _, jr in job_df.iterrows():
        job_text = str(jr.get(job_desc_col, ""))
        job_texts.append(job_text)

    job_embs = embedder.encode(job_texts)

    for idx, (_, jr) in enumerate(job_df.iterrows()):
        job_rows.append({
            "job_idx": int(jr.name),
            "job_title": jr.get("job_title", ""),
            "company": jr.get("company", ""),
            "job_emb": job_embs[idx]
        })

    pair_rows = []
    topk_rows = []

    for _, rr in resume_df.iterrows():
        resume_id = rr.get("ID", rr.name)
        resume_text = str(rr.get(resume_text_col, ""))

        resume_emb = embedder.encode(resume_text)[0]

        scored_rows = []

        for j in job_rows:
            semantic_sim = float(np.dot(resume_emb, j["job_emb"]))

            row = {
                "resume_id": resume_id,
                "job_idx": j["job_idx"],
                "job_title": j["job_title"],
                "company": j["company"],
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