"""Matching utilities for resume-job matching.

This module contains dictionary-based scoring helpers that are reusable across
notebooks and app code.
"""

from __future__ import annotations

import ast
import heapq
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

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


def normalize_text(text: Any) -> str:
    """Normalize free text for matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_skill_list(value: Any) -> List[str]:
    """Normalize a value into a list of non-empty, lowercase skill strings."""
    if isinstance(value, list):
        return [s.strip().lower() for s in value if isinstance(s, str) and s.strip()]

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        # JSON list string
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [s.strip().lower() for s in parsed if isinstance(s, str) and s.strip()]
        except Exception:
            pass

        # Python-style list string
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [s.strip().lower() for s in parsed if isinstance(s, str) and s.strip()]
        except Exception:
            pass

        # Fallback: comma/semicolon separated text
        if "," in value or ";" in value:
            parts = re.split(r"[;,]", value)
            return [p.strip().lower() for p in parts if p.strip()]

    return []


def compute_title_similarity(resume_category: Any, job_title: Any) -> float:
    """
    Compute a title similarity score in [0, 1] using:
    - exact/substr match
    - category alias map
    - fuzzy fallback
    """
    rc = normalize_text(resume_category)
    jt = normalize_text(job_title)

    if not rc or not jt:
        return 0.0

    if rc == jt:
        return 1.0

    if rc in jt or jt in rc:
        return 0.9

    for category_key, aliases in ROLE_SYNONYMS.items():
        if rc == normalize_text(category_key):
            for alias in aliases:
                alias_n = normalize_text(alias)
                if alias_n and alias_n in jt:
                    return 0.85

    rc_tokens = set(rc.split())
    jt_tokens = set(jt.split())
    token_overlap = len(rc_tokens & jt_tokens) / len(rc_tokens | jt_tokens) if (rc_tokens | jt_tokens) else 0.0
    fuzzy_score = SequenceMatcher(None, rc, jt).ratio()

    return round(max(token_overlap, fuzzy_score), 4)


def dictionary_compare(resume_skills: Any, job_skills: Any) -> Dict[str, Any]:
    """Compute dictionary-based match metrics for one resume-job pair."""
    resume_set = set(to_skill_list(resume_skills))
    job_set = set(to_skill_list(job_skills))

    if not job_set:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "extra_skills": [],
            "job_coverage": 0.0,
            "skill_jaccard": 0.0,
            "dictionary_score": 0.0,
        }

    matched = sorted(resume_set & job_set)
    missing = sorted(job_set - resume_set)
    extra = sorted(resume_set - job_set)

    job_coverage = len(matched) / len(job_set)
    jaccard = len(matched) / len(resume_set | job_set) if (resume_set | job_set) else 0.0

    # weighted dictionary score on 0-100 scale
    score = 0.7 * job_coverage + 0.3 * jaccard

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "job_coverage": round(job_coverage, 4),
        "skill_jaccard": round(jaccard, 4),
        "dictionary_score": round(score * 100, 2),
    }


def compute_dictionary_pairs_and_topk(
    resume_df: pd.DataFrame,
    job_df: pd.DataFrame,
    resume_skill_col: str = "extracted_skills_ranked",
    job_skill_col: str = "extracted_skills_ranked",
    top_k: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all resume-job pair scores and Top-K jobs per resume.

    """
    for col_name, df_name, df in (
        (resume_skill_col, "resume_df", resume_df),
        (job_skill_col, "job_df", job_df),
    ):
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found in {df_name}")


    job_rows: List[Dict[str, Any]] = []
    for _, jr in job_df.iterrows():
        job_rows.append(
            {
                "job_idx": int(jr.name),
                "job_skills": to_skill_list(jr.get(job_skill_col, [])),
                "job_link": jr.get("job_link", ""),
                "job_title": jr.get("job_title", ""),
                "company": jr.get("company", ""),
                "job_location": jr.get("job_location", ""),
            }
        )

    pair_rows: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []

    for _, rr in resume_df.iterrows():
        resume_id = rr.get("ID", rr.name)
        resume_category = rr.get("Category", "")
        resume_skills = to_skill_list(rr.get(resume_skill_col, []))

        scored_rows: List[Dict[str, Any]] = []
        for j in job_rows:
            comp = dictionary_compare(resume_skills, j["job_skills"])

            row = {
                "resume_id": resume_id,
                "resume_idx": int(rr.name),
                "resume_category": resume_category,
                "job_idx": j["job_idx"],
                "job_link": j["job_link"],
                "job_title": j["job_title"],
                "company": j["company"],
                "job_location": j["job_location"],

                # component scores
                "dictionary_score": comp["dictionary_score"],
                "job_coverage": comp["job_coverage"],
                "skill_jaccard": comp["skill_jaccard"],

                # skill breakdown
                "matched_skills": comp["matched_skills"],
                "missing_skills": comp["missing_skills"],
                "extra_skills": comp["extra_skills"],
            }

            pair_rows.append(row)
            scored_rows.append(row)

        best = heapq.nlargest(top_k, scored_rows, key=lambda x: x["dictionary_score"])
        for rank, entry in enumerate(best, start=1):
            out = entry.copy()
            out["rank"] = rank
            topk_rows.append(out)

    return pd.DataFrame(pair_rows), pd.DataFrame(topk_rows)


def dumps_skill_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return a copy with selected list-like columns serialized to JSON strings."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].apply(json.dumps)
    return out