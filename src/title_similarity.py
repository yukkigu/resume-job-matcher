from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


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


def normalize_title(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\\s]", " ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t


def compute_title_similarity(resume_title: Any, job_title: Any) -> float:
    """
    Compute a title similarity score in [0, 1] using:
    - exact/substr match
    - role synonym map
    - token overlap vs fuzzy ratio fallback
    """
    rt = normalize_title(resume_title)
    jt = normalize_title(job_title)
    if not rt or not jt:
        return 0.0

    if rt == jt:
        return 1.0
    if rt in jt or jt in rt:
        return 0.9

    for role_key, aliases in ROLE_SYNONYMS.items():
        if rt == normalize_title(role_key):
            for alias in aliases:
                if normalize_title(alias) and normalize_title(alias) in jt:
                    return 0.85

    rt_tokens = set(rt.split())
    jt_tokens = set(jt.split())
    token_overlap = len(rt_tokens & jt_tokens) / len(rt_tokens | jt_tokens) if (rt_tokens | jt_tokens) else 0.0
    fuzzy_score = SequenceMatcher(None, rt, jt).ratio()
    return round(max(token_overlap, fuzzy_score), 4)


def compute_title_similarity_df(
    resume_title: str,
    jobs_df: pd.DataFrame,
    job_title_col: str = "job_title",
    job_index_col: Optional[str] = None,
) -> pd.DataFrame:
    if job_title_col not in jobs_df.columns:
        raise KeyError(f"Column '{job_title_col}' not found in jobs_df")

    rows: List[Dict[str, object]] = []
    for idx, row in jobs_df.fillna("").iterrows():
        job_idx = int(row.get(job_index_col)) if job_index_col and job_index_col in jobs_df.columns else int(idx)
        title = str(row.get(job_title_col, ""))
        rows.append(
            {
                "job_index": job_idx,
                "job_title": title,
                "title_similarity": compute_title_similarity(resume_title, title),
            }
        )

    df = pd.DataFrame(rows).sort_values("title_similarity", ascending=False)
    df.insert(0, "rank_by_title", range(1, len(df) + 1))
    return df


def save_title_similarity_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out

