import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Ensure modules in ./src are importable as top-level modules (sentiment_analysis.py expects this)
SRC_PATH = (Path(__file__).parent / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import json

from dictionary_match import dictionary_compare, to_skill_list
from sentiment_analysis import ResumeJobMatcher
from skills import DEFAULT_SYNONYM_MAP, SkillVocabularyBuilder
from skills_extraction import SkillExtractor
from title_similarity import compute_title_similarity, compute_title_similarity_df, save_title_similarity_csv


st.set_page_config(page_title="Resume → Job Recommender", layout="wide")
st.title("Resume → Job Recommender")


@dataclass(frozen=True)
class JobIndex:
    titles: List[str]
    companies: List[str]
    job_clean_texts: List[str]
    job_skill_sets: List[Set[str]]
    job_keyword_sets: List[Set[str]]
    job_required_labels: List[Set[str]]
    vocab_keys: List[str]


def _extract_keyword_set(text: str, top_k: int = 25) -> Set[str]:
    cleaned = ResumeJobMatcher.clean_text(text)
    if not cleaned:
        return set()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#./-]{1,}", cleaned)
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "have",
        "has",
        "are",
        "was",
        "will",
        "your",
        "our",
        "you",
        "job",
        "role",
        "work",
        "team",
        "using",
        "use",
        "required",
        "preferred",
        "experience",
        "years",
        "year",
        "ability",
        "skills",
        "skill",
        "candidate",
        "resume",
        "description",
        "strong",
        "plus",
        "including",
        "etc",
    }
    filtered = [tok for tok in tokens if tok not in stopwords and len(tok) > 2]
    counts: Dict[str, int] = {}
    for tok in filtered:
        counts[tok] = counts.get(tok, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return {w for w, _ in top}


def _extract_skills_from_text(
    builder: SkillVocabularyBuilder,
    text: str,
    vocabulary: Set[str],
    max_ngram: int = 4,
) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    cleaned = ResumeJobMatcher.clean_text(text)
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+#./\\-]*", cleaned)

    found: List[str] = []
    for n in range(1, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            phrase_n = builder.normalize_skill(phrase)
            if not builder.is_valid_skill_phrase(phrase_n):
                continue
            canonical = builder.canonicalize(phrase_n)
            if canonical in vocabulary:
                found.append(canonical)

    for item in builder.parse_skills(text):
        item_n = builder.normalize_skill(item)
        if not builder.is_valid_skill_phrase(item_n):
            continue
        canonical = builder.canonicalize(item_n)
        if canonical in vocabulary:
            found.append(canonical)

    return list(dict.fromkeys(found))


def _job_required_labels(job_clean_text: str) -> Set[str]:
    required: Set[str] = set()
    for label, patterns in ResumeJobMatcher.DEFAULT_REQUIRED_QUAL_PATTERNS.items():
        if any(re.search(pattern, job_clean_text) for pattern in patterns):
            required.add(label)
    return required


@st.cache_data(show_spinner=False)
def load_jobs_and_vocab(min_frequency: int = 3) -> JobIndex:
    jobs_df, _ = ResumeJobMatcher.load_processed_data()

    vocab = ResumeJobMatcher.build_skill_vocabulary_from_jobs(
        jobs_df,
        skills_col="job_skills_cleaned",
        min_frequency=min_frequency,
    )
    vocab_keys = list(vocab.keys())

    builder = SkillVocabularyBuilder(synonym_map=DEFAULT_SYNONYM_MAP)
    vocab_set = set(vocab_keys) | set(builder.synonym_map.values())

    titles: List[str] = []
    companies: List[str] = []
    job_clean_texts: List[str] = []
    job_skill_sets: List[Set[str]] = []
    job_keyword_sets: List[Set[str]] = []
    job_requireds: List[Set[str]] = []

    extracted_path = (Path(__file__).parent / "data" / "processed" / "job_skills_extracted.csv").resolve()
    extractor = SkillExtractor(vocabulary=vocab, synonym_map=DEFAULT_SYNONYM_MAP, min_vocab_frequency=1)

    if extracted_path.exists():
        extracted_df = pd.read_csv(extracted_path).fillna("")
        if "job_index" not in extracted_df.columns or "extracted_skills" not in extracted_df.columns:
            raise KeyError(
                f"Expected columns ['job_index','extracted_skills'] in {extracted_path.name}, "
                f"got {extracted_df.columns.tolist()}"
            )
        extracted_skills_by_index: Dict[int, List[str]] = {}
        for _, row in extracted_df.iterrows():
            try:
                jidx = int(row.get("job_index"))
            except Exception:
                continue
            extracted_skills_by_index[jidx] = to_skill_list(row.get("extracted_skills", ""))
    else:
        extracted_skills_by_index = {}

    for _, row in jobs_df.fillna("").iterrows():
        job_index = int(row.name)
        titles.append(str(row.get("job_title", "")))
        companies.append(str(row.get("company", "")))
        combined = str(row.get("job_combined_text", "")) or " ".join(
            [
                str(row.get("job_title", "")),
                str(row.get("company", "")),
                str(row.get("job_summary_cleaned", "")),
                str(row.get("job_skills_cleaned", "")),
            ]
        )
        clean = ResumeJobMatcher.clean_text(combined)
        job_clean_texts.append(clean)

        # Prefer extracted skills from job_skills_extracted.csv; otherwise compute and save later.
        extracted_skills = extracted_skills_by_index.get(job_index)
        if extracted_skills is None:
            extracted_skills = extractor.extract_from_text(str(row.get("job_skills_cleaned", "")))
            extracted_skills_by_index[job_index] = extracted_skills
        job_skill_sets.append(set(extracted_skills))

        job_keyword_sets.append(_extract_keyword_set(clean, top_k=25))
        job_requireds.append(_job_required_labels(clean))

    if not extracted_path.exists():
        extracted_path.parent.mkdir(parents=True, exist_ok=True)
        out_rows = []
        for jidx, skills in extracted_skills_by_index.items():
            out_rows.append(
                {
                    "job_index": int(jidx),
                    "extracted_skills": json.dumps(list(skills)),
                }
            )
        pd.DataFrame(out_rows).sort_values("job_index").to_csv(extracted_path, index=False)

    return JobIndex(
        titles=titles,
        companies=companies,
        job_clean_texts=job_clean_texts,
        job_skill_sets=job_skill_sets,
        job_keyword_sets=job_keyword_sets,
        job_required_labels=job_requireds,
        vocab_keys=vocab_keys,
    )


def _cosine_sim_matrix(job_embeddings: np.ndarray, resume_embedding: np.ndarray) -> np.ndarray:
    # Works whether embeddings are normalized or not.
    resume = resume_embedding.reshape(1, -1)
    job_norm = np.linalg.norm(job_embeddings, axis=1, keepdims=True) + 1e-12
    resume_norm = np.linalg.norm(resume, axis=1, keepdims=True) + 1e-12
    return (job_embeddings @ resume.T).ravel() / (job_norm.ravel() * resume_norm.ravel())


@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def embed_jobs(model_name: str, job_texts: Sequence[str]) -> np.ndarray:
    model = load_sentence_transformer(model_name)
    # normalize so cosine similarity becomes dot product
    return model.encode(list(job_texts), convert_to_numpy=True, normalize_embeddings=True)


def _rank_with_composite(
    resume_text: str,
    index: JobIndex,
    model_name: str,
    top_k: int,
    resume_title: str,
    title_weight: float,
    candidate_pool: int = 250,
) -> pd.DataFrame:
    resume_clean = ResumeJobMatcher.clean_text(resume_text)
    if not resume_clean:
        return pd.DataFrame()

    # ----- semantic similarity (fast path if transformer loads) -----
    semantic_scores: Optional[np.ndarray] = None
    semantic_backend = "sentence-transformers"
    try:
        model = load_sentence_transformer(model_name)
        resume_emb = model.encode([resume_clean], convert_to_numpy=True, normalize_embeddings=True)[0]
        job_embs = embed_jobs(model_name, index.job_clean_texts)
        semantic_scores = (job_embs @ resume_emb).ravel()
    except Exception:
        semantic_backend = "tfidf"

    if semantic_scores is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([resume_clean] + list(index.job_clean_texts))
        semantic_scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    # take a candidate pool by semantic similarity, then compute full composite score
    pool = int(min(max(candidate_pool, top_k), len(semantic_scores)))
    candidate_idx = np.argsort(semantic_scores)[::-1][:pool]

    matcher = ResumeJobMatcher(vocabulary=index.vocab_keys, model_name=model_name)
    extractor = SkillExtractor(vocabulary={k: 1 for k in index.vocab_keys}, synonym_map=DEFAULT_SYNONYM_MAP)
    resume_skills = set(extractor.extract_from_text(resume_text))
    resume_keywords = _extract_keyword_set(resume_clean, top_k=25)

    resume_has_label: Dict[str, bool] = {}
    for label, patterns in matcher.qual_patterns.items():
        resume_has_label[label] = any(re.search(p, resume_clean) for p in patterns)

    rows: List[Dict[str, object]] = []
    for idx in candidate_idx:
        job_clean = index.job_clean_texts[int(idx)]

        job_skills = index.job_skill_sets[int(idx)]
        overlap = matcher.jaccard_similarity(resume_skills, job_skills)

        job_keywords = index.job_keyword_sets[int(idx)]
        keyword_score = (len(resume_keywords & job_keywords) / len(job_keywords)) if job_keywords else 0.0

        required_labels = index.job_required_labels[int(idx)]
        if not required_labels:
            qualification_score = 1.0
            missing_quals: List[str] = []
        else:
            missing_quals = [lab for lab in sorted(required_labels) if not resume_has_label.get(lab, False)]
            qualification_score = 1.0 - (len(missing_quals) / len(required_labels))

        semantic = float(semantic_scores[int(idx)])
        overall = (
            matcher.weights["semantic_similarity"] * semantic
            + matcher.weights["skill_overlap"] * overlap
            + matcher.weights["keyword_relevance"] * keyword_score
            + matcher.weights["qualification_match"] * qualification_score
        )
        title_sim = compute_title_similarity(resume_title, index.titles[int(idx)]) if resume_title else 0.0
        overall_with_title = (1.0 - title_weight) * overall + title_weight * title_sim

        rows.append(
            {
                "job_index": int(idx),
                "title": index.titles[int(idx)],
                "company": index.companies[int(idx)],
                "overall_score": round(float(overall_with_title) * 100.0, 2),
                "semantic_similarity": round(float(semantic), 4),
                "skill_overlap_score": round(float(overlap), 4),
                "keyword_relevance": round(float(keyword_score), 4),
                "qualification_match": round(float(qualification_score), 4),
                "title_similarity": round(float(title_sim), 4),
                "matched_skills_preview": ", ".join(sorted(list(resume_skills & job_skills))[:10]),
                "missing_skills_preview": ", ".join(sorted(list(job_skills - resume_skills))[:10]),
                "missing_qualifications": ", ".join(missing_quals[:10]),
            }
        )

    df = pd.DataFrame(rows).sort_values("overall_score", ascending=False).head(top_k)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.attrs["semantic_backend"] = semantic_backend
    return df


def _rank_with_skills_tfidf_hybrid(
    resume_text: str,
    index: JobIndex,
    top_k: int,
    dict_weight: float,
    tfidf_weight: float,
    resume_title: str,
    title_weight: float,
) -> pd.DataFrame:
    resume_clean = ResumeJobMatcher.clean_text(resume_text)
    if not resume_clean:
        return pd.DataFrame()

    extractor = SkillExtractor(vocabulary={k: 1 for k in index.vocab_keys}, synonym_map=DEFAULT_SYNONYM_MAP)
    resume_skills = extractor.extract_from_text(resume_text)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([resume_clean] + list(index.job_clean_texts))
    tfidf_scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    rows: List[Dict[str, object]] = []
    for job_index in range(len(index.job_clean_texts)):
        job_skills = sorted(list(index.job_skill_sets[job_index]))
        dict_result = dictionary_compare(resume_skills, job_skills)
        dict_score = float(dict_result["dictionary_score"]) / 100.0
        tfidf_sim = float(tfidf_scores[job_index])
        hybrid = (dict_weight * dict_score) + (tfidf_weight * tfidf_sim)
        title_sim = compute_title_similarity(resume_title, index.titles[job_index]) if resume_title else 0.0
        hybrid_with_title = (1.0 - title_weight) * hybrid + title_weight * title_sim
        rows.append(
            {
                "job_index": int(job_index),
                "title": index.titles[job_index],
                "company": index.companies[job_index],
                "hybrid_score": round(float(hybrid_with_title), 6),
                "dictionary_score": round(float(dict_score) * 100.0, 2),
                "tfidf_similarity": round(float(tfidf_sim), 6),
                "title_similarity": round(float(title_sim), 4),
                "matched_skills_preview": ", ".join(dict_result.get("matched_skills", [])[:10]),
                "missing_skills_preview": ", ".join(dict_result.get("missing_skills", [])[:10]),
            }
        )
    df = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(top_k)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K jobs", min_value=1, max_value=20, value=5, step=1)
    min_freq = st.slider("Skill vocab min frequency", min_value=1, max_value=25, value=3, step=1)
    resume_title = st.text_input("Resume title / target role (optional)", value="")
    title_weight = st.slider("Title similarity weight", 0.0, 0.5, 0.15, 0.05)
    method = st.selectbox(
        "Method",
        [
            "MPNet (composite scoring)",
            "MiniLM (composite scoring)",
            "BGE-small (composite scoring)",
            "Skills + TF-IDF (hybrid)",
        ],
    )

    if method == "Skills + TF-IDF (hybrid)":
        dict_weight = st.slider("Dictionary weight", 0.0, 1.0, 0.6, 0.05)
        tfidf_weight = 1.0 - dict_weight
        st.caption(f"TF-IDF weight = {tfidf_weight:.2f}")
    else:
        candidate_pool = st.slider("Candidate pool (speed/quality)", 50, 1000, 250, 50)

    st.divider()
    st.caption("Models may download on first run (SentenceTransformers).")


index = load_jobs_and_vocab(min_frequency=min_freq)

resume_text = st.text_area(
    "Paste resume text",
    height=260,
    placeholder="Paste resume text here (like the Resume_str column).",
)

col_a, col_b = st.columns([1, 2])
with col_a:
    run = st.button("Get recommendations", type="primary")
with col_b:
    st.caption(f"Jobs indexed: {len(index.job_clean_texts)}")


if run:
    if not resume_text.strip():
        st.warning("Paste a resume first.")
        st.stop()

    with st.spinner("Ranking jobs..."):
        try:
            if method == "MPNet (composite scoring)":
                df = _rank_with_composite(
                    resume_text=resume_text,
                    index=index,
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    top_k=top_k,
                    resume_title=resume_title,
                    title_weight=title_weight,
                    candidate_pool=candidate_pool,
                )
                backend = df.attrs.get("semantic_backend", "unknown")
                st.caption(f"Semantic backend: `{backend}`")
            elif method == "MiniLM (composite scoring)":
                df = _rank_with_composite(
                    resume_text=resume_text,
                    index=index,
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    top_k=top_k,
                    resume_title=resume_title,
                    title_weight=title_weight,
                    candidate_pool=candidate_pool,
                )
                backend = df.attrs.get("semantic_backend", "unknown")
                st.caption(f"Semantic backend: `{backend}`")
            elif method == "BGE-small (composite scoring)":
                df = _rank_with_composite(
                    resume_text=resume_text,
                    index=index,
                    model_name="BAAI/bge-small-en-v1.5",
                    top_k=top_k,
                    resume_title=resume_title,
                    title_weight=title_weight,
                    candidate_pool=candidate_pool,
                )
                backend = df.attrs.get("semantic_backend", "unknown")
                st.caption(f"Semantic backend: `{backend}`")
            else:
                df = _rank_with_skills_tfidf_hybrid(
                    resume_text=resume_text,
                    index=index,
                    top_k=top_k,
                    dict_weight=dict_weight,
                    tfidf_weight=tfidf_weight,
                    resume_title=resume_title,
                    title_weight=title_weight,
                )
        except ModuleNotFoundError as exc:
            if "torchvision" in str(exc):
                st.error(
                    "Missing dependency `torchvision`. Install it (recommended) or switch to "
                    "`Skills + TF-IDF (hybrid)`."
                )
                st.code("python3 -m pip install torchvision")
                st.info("Falling back to `Skills + TF-IDF (hybrid)` for this run.")
                df = _rank_with_skills_tfidf_hybrid(
                    resume_text=resume_text,
                    index=index,
                    top_k=top_k,
                    dict_weight=0.6,
                    tfidf_weight=0.4,
                    resume_title=resume_title,
                    title_weight=title_weight,
                )
            else:
                raise

    if df.empty:
        st.warning("No results. Is the resume text empty after cleaning?")
        st.stop()

    st.subheader("Top recommendations")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if resume_title.strip():
        # Save title similarity for all jobs as a CSV, and offer download.
        jobs_df, _ = ResumeJobMatcher.load_processed_data()
        ts_df = compute_title_similarity_df(resume_title.strip(), jobs_df, job_title_col="job_title")
        out_path = save_title_similarity_csv(ts_df, Path("outputs") / "title_similarity_scores.csv")

        with st.expander("Title similarity (CSV)"):
            st.caption(f"Saved to `{out_path}`")
            st.dataframe(ts_df.head(20), use_container_width=True, hide_index=True)
            st.download_button(
                "Download title similarity CSV",
                data=out_path.read_bytes(),
                file_name=out_path.name,
                mime="text/csv",
            )

    with st.expander("Show raw resume preview"):
        cleaned = ResumeJobMatcher.clean_text(resume_text)
        st.text((cleaned[:2000] + "…") if len(cleaned) > 2000 else cleaned)
