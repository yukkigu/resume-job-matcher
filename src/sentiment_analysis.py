"""Resume/job matching module.

Despite the filename, this module implements semantic resume-to-job matching
for the final project proposal described by the user.

Features
- Text cleaning and normalization
- Skill extraction with dictionary + synonym normalization
- BERT/sentence-transformer embeddings when available
- TF-IDF fallback when transformer models are unavailable
- Composite fit score using:
    * skill overlap
    * semantic similarity
    * keyword relevance
    * qualification match
- Template-based recommendation summary
- Top-k job recommendation helper

This file is self-contained and designed to be easy to integrate into a larger
pipeline that reads the Kaggle resume and LinkedIn jobs datasets.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from skills import SkillVocabularyBuilder, DEFAULT_SYNONYM_MAP 

import numpy as np
import pandas as pd 
from pathlib import Path 
import os

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity 
except Exception as exc: # pragma: no cover
    raise ImportError(
        "scikit-learn is required for sentiment_analysis.py to run."
    ) from exc

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

JOBS_CSV = DATA_DIR / "jobs_subset.csv"
RESUMES_CSV = DATA_DIR / "resumes_subset.csv"


@dataclass
class MatchBreakdown:
    overall_score: float
    semantic_similarity: float
    skill_overlap_score: float
    keyword_relevance: float 
    qualification_match: float
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    missing_qualifications: List[str] = field(default_factory=list)
    summary: str = ""

class TextEmbedder:
    """Embeds texts with sentence-transformers if available, TF-IDF otherwise."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.backend = "tfidf"
        self._load_transformer_model()
    
    def _load_transformer_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
        except Exception:
            self._model = None
            self.backend = "tfidf"
    
    def cosine_sim(self, first_text: str, second_text: str) -> float:
        if not first_text.strip() or not second_text.strip():
            return 0.0
        
        if self._model is not None:
            try:
                embeddings = self._model.encode([first_text, second_text], convert_to_numpy=True)
                first, second = embeddings[0], embeddings[1]
                denom = np.linalg.norm(first) * np.linalg.norm(second)
                if denom == 0:
                    return 0.0
                return float(np.dot(first, second) / denom)
            except Exception:
                pass
    
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vec.fit_transform([first_text, second_text])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])

class ResumeJobMatcher:
    DEFAULT_REQUIRED_QUAL_PATTERNS = {
        "bachelor's degree": [
            r"\bbachelor'?s degree\b",
            r"\bbs\b",
            r"\bba\b",
            r"\bundergraduate degree\b",
        ],
        "master's degree": [
            r"\bmaster'?s degree\b",
            r"\bms\b",
            r"\bma\b",
            r"\bm\.s\.\b",
        ],
        "phd": [
            r"\bph\.?d\b",
            r"\bdoctorate\b",
        ],
        "sql": [
            r"\bsql\b",
            r"\bpostgresql\b",
            r"\bmysql\b",
            r"\bsql server\b",
        ],
        "python": [
            r"\bpython\b",
            r"\bpython 3\b",
        ],
        "machine learning": [
            r"\bmachine learning\b",
            r"\bml\b",
        ],
        "communication skills": [
            r"\bcommunication skills\b",
            r"\bstrong communication\b",
            r"\bexcellent communication\b",
        ],
        "leadership": [
            r"\bleadership\b",
            r"\blead\b",
            r"\bteam leadership\b",
            r"\bmanag(e|ing|ement)\b",
        ],
        "3+ years experience": [
            r"\b3\+?\s+years? of experience\b",
            r"\bminimum 3 years\b",
            r"\bat least 3 years\b",
        ],
        "5+ years experience": [
            r"\b5\+?\s+years? of experience\b",
            r"\bminimum 5 years\b",
            r"\bat least 5 years\b",
        ],
    }

    NOISE_SKILLS = {
        "a", "an", "the", "be", "work", "skills", "skill", "experience",
        "knowledge", "general", "office", "support", "review", "service",
        "reports", "reporting", "documentation", "education"
    }

    BENEFIT_PATTERNS = [
        r"\b401k\b",
        r"\b401\(k\)\b",
        r"\bsalary\b",
        r"\bcompetitive salary\b",
        r"\bbonus\b",
        r"\bmedical\b",
        r"\bdental\b",
        r"\bvision\b",
        r"\bbenefits\b",
        r"\bpaid time off\b",
        r"\bpto\b",
        r"\blife insurance\b",
        r"\bretirement\b",
        r"\bwellness\b",
        r"\bremote\b",
        r"\bhybrid\b",
    ]

    def __init__(
            self,
            synonym_map: Optional[Dict[str, str]] = None,
            vocabulary: Optional[Iterable[str]] = None,
            weights: Optional[Dict[str, float]] = None,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            qual_patterns = None
    ) -> None:
        self.qual_patterns = qual_patterns or self.DEFAULT_REQUIRED_QUAL_PATTERNS
        self.skill_builder = SkillVocabularyBuilder(synonym_map or DEFAULT_SYNONYM_MAP)
        self.synonym_map = self.skill_builder.synonym_map
        self.vocabulary = {
            self.skill_builder.canonicalize(skill)
            for skill in (vocabulary or [])
            if self.skill_builder.is_valid_skill_phrase(self.skill_builder.normalize_skill(skill))
        }
        self.embedder = TextEmbedder(model_name=model_name)
        self.weights = weights or {
            "semantic_similarity": 0.35,
            "skill_overlap": 0.35,
            "keyword_relevance": 0.15,
            "qualification_match": 0.15,
        }
        self._validate_weights()
    
    def is_meaningful_skill(self, skill: str) -> bool:
        skill = self.skill_builder.normalize_skill(skill)

        if not skill or skill in self.NOISE_SKILLS:
            return False

        if not self.skill_builder.is_valid_skill_phrase(skill):
            return False

        for pattern in self.BENEFIT_PATTERNS:
            if re.search(pattern, skill):
                return False

        # reject very short generic single words
        allow_short = {"c", "c#", "c++", "r", "go", "sql", "aws", "gcp", "api", "etl", "bi"}
        if skill not in allow_short and len(skill.split()) == 1 and len(skill) <= 3:
            return False

        return True
    
    @staticmethod
    def clean_text(text: str) -> str:
        if text is None:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9+.#/\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    @staticmethod
    def _deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
        return list(dict.fromkeys(items))
    
    def _extract_candidate_skill_phrases(self, text: str) -> List[str]:
        """Create candidate skill phrases from free text using the same normalization rules."""
        if not text:
            return []
        
        normalized_text = self.clean_text(text)
        candidates: List[str] = []

        #1- to 4-gram phrase candidates from text 
        tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+#./\-]*", normalized_text)
        for n in range(1, 5):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                phrase = self.skill_builder.normalize_skill(phrase)
                if self.skill_builder.is_valid_skill_phrase(phrase):
                    candidates.append(phrase)

        # Also parse delimiter-like lists if present in the text
        parsed = self.skill_builder.parse_skills(text)
        for item in parsed:
            phrase = self.skill_builder.normalize_skill(item)
            if self.skill_builder.is_valid_skill_phrase(phrase):
                candidates.append(phrase)
        return self._deduplicate_preserve_order(candidates)
    
    def extract_skills(self, text: str) -> Set[str]:
        """Extract canonical skills from free text using SkillVocabularyBuilder utilities."""
        candidates = self._extract_candidate_skill_phrases(text)
        found = set()

        for phrase in candidates:
            normalized = self.skill_builder.normalize_skill(phrase)
            canonical = self.skill_builder.canonicalize(normalized)

            if not self.is_meaningful_skill(canonical):
                continue

            if canonical in self.vocabulary or canonical in self.synonym_map.values():
                found.add(canonical)

        return found
    
    def extracts_keywords(self, text: str, top_k: int = 20) -> List[str]:
        cleaned = self.clean_text(text)
        if not cleaned: 
            return []
        tokens = re.findall(r"[a-zA-Z][a-zA-Z+#./-]{1,}", cleaned)
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "have", "has",
            "are", "was", "will", "your", "our", "you", "job", "role", "work",
            "team", "using", "use", "required", "preferred", "experience", "years",
            "year", "ability", "skills", "skill", "candidate", "resume", "description",
            "strong", "plus", "including", "etc",
        }
        filtered = [tok for tok in tokens if tok not in stopwords and len(tok) > 2]
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(top_k)]
    
    @staticmethod
    def jaccard_similarity(first_set: Set[str], second_set: Set[str]) -> float:
        if not first_set and not second_set:
            return 0.0
        union = first_set | second_set
        if not union:
            return 0.0
        return len(first_set & second_set) / len(union)
    
    def keywords_relevance_score(self, resume_text: str, job_text: str) -> float:
        resume_keywords = set(self.extracts_keywords(resume_text, top_k=25))
        job_keywords = set(self.extracts_keywords(job_text, top_k=25))
        if not job_keywords:
            return 0.0
        return len(resume_keywords & job_keywords) / len(job_keywords)

    def qualification_match_score(self, resume_text: str, job_text: str) -> Tuple[float, List[str]]:
        resume_clean = self.clean_text(resume_text)
        job_clean = self.clean_text(job_text)

        present_requirements: List[str] = []
        missing_requirements: List[str] = []

        for label, patterns in self.qual_patterns.items():
            job_has_req = any(re.search(pattern, job_clean) for pattern in patterns)
            if not job_has_req:
                continue

            present_requirements.append(label)

            resume_has_req = any(re.search(pattern, resume_clean) for pattern in patterns)
            if not resume_has_req:
                missing_requirements.append(label)

        if not present_requirements:
            return 1.0, []

        score = 1.0 - (len(missing_requirements) / len(present_requirements))
        return max(0.0, score), missing_requirements
    
    def compute_similarity(self, resume_text: str, job_text: str) -> MatchBreakdown:
        resume_clean = self.clean_text(resume_text)
        job_clean = self.clean_text(job_text)
        
        resume_skills = self.extract_skills(resume_text)
        job_skills = self.extract_skills(job_text)
        matched_skills = sorted(
            skill for skill in (resume_skills & job_skills)
            if self.is_meaningful_skill(skill)
        )

        missing_skills = sorted(
            skill for skill in (job_skills - resume_skills)
            if self.is_meaningful_skill(skill)
        )
        #matched_skills = sorted(resume_skills & job_skills)
        #missing_skills = sorted(job_skills - resume_skills)

        semantic_score = self.embedder.cosine_sim(resume_clean, job_clean)
        overlap_score = self.jaccard_similarity(resume_skills, job_skills)
        keyword_score = self.keywords_relevance_score(resume_clean, job_clean)
        qualification_score, missing_qualifications = self.qualification_match_score(
            resume_clean, job_clean
        )

        overall = (
            self.weights["semantic_similarity"] * semantic_score
            + self.weights["skill_overlap"] * overlap_score
            + self.weights["keyword_relevance"] * keyword_score
            + self.weights["qualification_match"] * qualification_score
        )
        overall_pct = round(overall * 100, 2)

        return MatchBreakdown(
            overall_score = overall_pct,
            semantic_similarity=round(semantic_score, 4),
            skill_overlap_score=round(overlap_score, 4),
            keyword_relevance=round(keyword_score, 4),
            qualification_match=round(qualification_score, 4),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            missing_qualifications=missing_qualifications,
            summary=self.generate_summary(
                overall_score=overall_pct,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                missing_qualifications=missing_qualifications,
            ),
        )

    def rank_jobs(
            self, 
            resume_text: str, 
            jobs: Sequence[Dict[str, str]], 
            job_text_fields: Sequence[str] = ("description", "skills", "title"),
            top_k = 5
    ) -> List[Dict[str, object]]:
        ranked: List[Dict[str, object]] = []
        for idx, job in enumerate(jobs):
            combined_text = " ".join(str(job.get(field, "")) for field in job_text_fields)
            breakdown = self.compute_similarity(resume_text, combined_text)
            ranked.append(
                {
                    "job_index": idx,
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "overall_score": breakdown.overall_score,
                    "matched_skills": breakdown.matched_skills,
                    "missing_skills": breakdown.missing_skills,
                    "summary": breakdown.summary,
                    "breakdown": breakdown,
                }
            )
        ranked.sort(key=lambda item: item["overall_score"], reverse=True)
        return ranked[:top_k]
    

    @staticmethod
    def generate_summary(
        overall_score: float,
        matched_skills: Sequence[str],
        missing_skills: Sequence[str],
        missing_qualifications: Sequence[str],
    ) -> str:
        if overall_score >= 80:
            fit = "strong"
        elif overall_score >= 60:
            fit = "moderate"
        else:
            fit = "limited"

        matched_preview = ", ".join(matched_skills[:5]) if matched_skills else "few directly matched skills"
        missing_preview = ", ".join(missing_skills[:5]) if missing_skills else "no major skill gaps"
        qual_preview = (
            "; missing qualifications include " + ", ".join(missing_qualifications)
            if missing_qualifications
            else ""
        )
        return (
            f"Candidate appears to be a {fit} fit ({overall_score:.1f}%). "
            f"Top matched skills: {matched_preview}. "
            f"Main gaps: {missing_preview}{qual_preview}."
        )

    def _validate_weights(self) -> None:
        required = {
            "semantic_similarity",
            "skill_overlap",
            "keyword_relevance",
            "qualification_match",
        }
        missing = required - set(self.weights)
        if missing:
            raise ValueError(f"Missing weight keys: {sorted(missing)}")
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


    def build_skill_vocabulary_from_jobs(
        df: pd.DataFrame,
        skills_col: str = "job_skills",
        synonym_map: Optional[Dict[str, str]] = None,
        min_frequency: int = 10,
    ) -> Dict[str, int]:
        builder = SkillVocabularyBuilder(synonym_map or DEFAULT_SYNONYM_MAP)
        builder.fit_from_dataframe(df, skills_col=skills_col)
        return builder.build_vocabulary(min_frequency=min_frequency)


    def compare_resume_to_job(
        resume_text: str,
        job_description: str,
        synonym_map: Optional[Dict[str, str]] = None,
        vocabulary: Optional[Iterable[str]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Dict[str, object]:
        matcher = ResumeJobMatcher(
            synonym_map=synonym_map,
            vocabulary=vocabulary,
            model_name=model_name,
        )
        result = matcher.compute_similarity(resume_text, job_description)
        return {
            "overall_score": result.overall_score,
            "semantic_similarity": result.semantic_similarity,
            "skill_overlap_score": result.skill_overlap_score,
            "keyword_relevance": result.keyword_relevance,
            "qualification_match": result.qualification_match,
            "matched_skills": result.matched_skills,
            "missing_skills": result.missing_skills,
            "missing_qualifications": result.missing_qualifications,
            "summary": result.summary,
            "embedding_backend": matcher.embedder.backend,
        }

    def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data" / "processed"

        jobs_path = data_dir / "jobs_subset.csv"
        resumes_path = data_dir / "resumes_subset.csv"

        if not jobs_path.exists():
            raise FileNotFoundError(f"Jobs file not found: {jobs_path}")
        if not resumes_path.exists():
            raise FileNotFoundError(f"Resumes file not found: {resumes_path}")

        jobs_df = pd.read_csv(jobs_path)
        resumes_df = pd.read_csv(resumes_path)

        print(f"[INFO] Loaded jobs: {jobs_df.shape}")
        print(f"[INFO] Loaded resumes: {resumes_df.shape}")

        return jobs_df, resumes_df


    def _find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(f"None of these columns were found: {list(candidates)}")

    @staticmethod
    def prepare_jobs_as_records(jobs_df: pd.DataFrame) -> List[Dict[str, str]]:
        title_col = ResumeJobMatcher._find_first_existing_column(jobs_df, ["job_title", "job_title_cleaned", "title"])
        company_col = ResumeJobMatcher._find_first_existing_column(jobs_df, ["company"])
        desc_col = ResumeJobMatcher._find_first_existing_column(jobs_df, ["job_combined_text", "job_summary", "job_summary_cleaned"])
        skills_col = ResumeJobMatcher._find_first_existing_column(jobs_df, ["job_skills", "job_skills_cleaned"])

        jobs: List[Dict[str, str]] = []
        for _, row in jobs_df.fillna("").iterrows():
            jobs.append(
                {
                    "title": str(row.get(title_col, "")),
                    "company": str(row.get(company_col, "")),
                    "description": str(row.get(desc_col, "")),
                    "skills": str(row.get(skills_col, "")),
                }
            )
        return jobs

    @staticmethod
    def prepare_resumes(resumes_df: pd.DataFrame) -> List[Dict[str, str]]:
        resume_col = ResumeJobMatcher._find_first_existing_column(
            resumes_df,
            ["resume_text", "resume", "text", "Resume_str", "Resume", "content"]
        )

        id_col = None
        for candidate in ["resume_id", "id", "ID"]:
            if candidate in resumes_df.columns:
                id_col = candidate
                break

        resumes: List[Dict[str, str]] = []
        for idx, row in resumes_df.fillna("").iterrows():
            resumes.append(
                {
                    "resume_id": str(row[id_col]) if id_col else str(idx),
                    "resume_text": str(row.get(resume_col, "")),
                }
            )
        return resumes
    
if __name__ == "__main__":

    jobs_df, resumes_df = ResumeJobMatcher.load_processed_data()

    # Build vocabulary from real job skills
    vocab = ResumeJobMatcher.build_skill_vocabulary_from_jobs(
        jobs_df,
        skills_col="job_skills_cleaned",   # change only if your CSV uses a different column name
        min_frequency=3,
    )

    print(f"[INFO] Vocabulary size: {len(vocab)}")

    matcher = ResumeJobMatcher(
        vocabulary=vocab.keys(),
        model_name="all-MiniLM-L6-v2",
    )

    print(f"[INFO] Embedding backend: {matcher.embedder.backend}")

    jobs = ResumeJobMatcher.prepare_jobs_as_records(jobs_df)
    resumes = ResumeJobMatcher.prepare_resumes(resumes_df)

    # Example: rank jobs for the first 3 resumes
    for resume in resumes[:3]:
        print("\n" + "=" * 80)
        print(f"Resume ID: {resume['resume_id']}")

        ranked = matcher.rank_jobs( 
            resume_text=resume["resume_text"],
            jobs=jobs,
            job_text_fields=("description", "skills", "title"),
            top_k=5,
        )

        for i, item in enumerate(ranked, start=1):
            print(f"\nTop {i}")
            print(f"Title: {item['title']}")
            print(f"Company: {item['company']}")
            print(f"Score: {item['overall_score']}")
            print(f"Matched skills: {item['matched_skills']}")
            print(f"Missing skills: {item['missing_skills'][:10]}")
            print(f"Summary: {item['summary']}")

'''
if __name__ == "__main__":
    sample_resume = """
    Data analyst with 3 years of experience using Python, SQL, Excel, Tableau,
    statistics, machine learning, and pandas. Built dashboards and automated
    reporting pipelines. Bachelor's degree in Computer Science.
    """

    sample_job = """
    We are hiring a Data Analyst with 2+ years experience. Required skills:
    Python, SQL, Tableau, Power BI, Excel, communication skills, and statistics.
    Bachelor's degree required. Experience with AWS is a plus.
    """

    demo_vocab = {
        "python", "sql", "excel", "tableau", "power bi", "statistics",
        "communication skills", "amazon web services", "machine learning", "pandas"
    }

    result = compare_resume_to_job(sample_resume, sample_job, vocabulary=demo_vocab)
    for key, value in result.items():
        print(f"{key}: {value}")
'''


    






