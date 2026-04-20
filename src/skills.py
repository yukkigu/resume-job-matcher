# Skill vocabulary building and normalization utilities

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Define a class to build a skill vocabulary from job skill text
class SkillVocabularyBuilder:
    """Builds a canonical skill vocabulary from job skill text."""

    def __init__(self, synonym_map: Optional[Dict[str, str]] = None):
        # synonym_map maps variant -> canonical, e.g. "py" -> "python"
        self.synonym_map = synonym_map or {}
        self.raw_frequency: Counter = Counter()
        self.canonical_frequency: Counter = Counter()

    @staticmethod
    def parse_skills(value) -> List[str]:
        """
        Parse mixed-format job skill values:
        - JSON list string
        - Python list string
        - delimiter-separated string
        """
        if pd.isna(value):
            return []

        s = str(value).strip()
        if not s:
            return []

        # Most rows in this dataset are plain delimiter-separated text,
        # so use structured parsing only when the value clearly looks like a list.
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        if s.startswith("("):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        # Fallback: split by common delimiters
        parts = re.split(r"[|,;/]", s)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """
        Normalize skill string while preserving important chars for tech tokens
        like c++, c#, node.js.
        """
        s = skill.lower().strip()
        s = re.sub(r"\s+", " ", s)
        # keep chars useful for technical/medical skills
        s = re.sub(r"[^\w\+\#\.\-\/\(\) ]", "", s)
        return s.strip()
    
    @staticmethod
    def is_valid_skill_phrase(skill: str) -> bool:
        """Filter out noisy non-skill phrases from raw job skill text."""
        if not skill:
            return False
        if len(skill) > 80:
            return False

        bad_prefixes = (
            "ability to ",
            "experience in ",
            "minimum ",
            "must be ",
            "required ",
            "strong ",
            "knowledge of ",
        )
        if skill.startswith(bad_prefixes):
            return False

        return True

    def canonicalize(self, skill: str) -> str:
        """Map normalized skill to canonical skill if synonym exists."""
        return self.synonym_map.get(skill, skill)

    def fit_from_series(self, skills_series: pd.Series) -> None:
        """Build frequency counters from a pandas Series of job skills."""
        self.raw_frequency.clear()
        self.canonical_frequency.clear()

        for value in skills_series:
            parsed = self.parse_skills(value)
            normalized = [self.normalize_skill(x) for x in parsed]
            normalized = [x for x in normalized if self.is_valid_skill_phrase(x)]
            normalized_unique = list(dict.fromkeys(normalized))  # preserve order, remove duplicates
            self.raw_frequency.update(normalized_unique)

        for skill, count in self.raw_frequency.items():
            canonical = self.canonicalize(skill)
            self.canonical_frequency[canonical] += count

    def fit_from_dataframe(self, df: pd.DataFrame, skills_col: str = "job_skills") -> None:
        """Build counters from a DataFrame column."""
        if skills_col not in df.columns:
            raise KeyError(f"Column '{skills_col}' not found in dataframe")
        self.fit_from_series(df[skills_col])

    def build_vocabulary(self, min_frequency: int = 10) -> Dict[str, int]:
        """Return canonical skill vocabulary filtered by frequency."""
        return {
            skill: int(count)
            for skill, count in self.canonical_frequency.items()
            if count >= min_frequency
        }

    def top_skills(self, n: int = 30) -> List[Tuple[str, int]]:
        """Return top-N canonical skills by frequency."""
        return self.canonical_frequency.most_common(n)

    def save_outputs(
        self,
        output_dir: str,
        min_frequency: int = 10,
        vocab_json_name: str = "skill_vocabulary.json",
        vocab_csv_name: str = "skill_vocabulary.csv",
        meta_json_name: str = "skill_vocabulary_meta.json",
    ) -> Dict[str, str]:
        """Save vocabulary, CSV frequency table, and metadata to output directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        vocab = self.build_vocabulary(min_frequency=min_frequency)

        vocab_json_path = out / vocab_json_name
        vocab_csv_path = out / vocab_csv_name
        meta_json_path = out / meta_json_name

        with open(vocab_json_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)

        rows = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        pd.DataFrame(rows, columns=["skill", "count"]).to_csv(vocab_csv_path, index=False)

        meta = {
            "total_skill_mentions_raw": int(sum(self.raw_frequency.values())),
            "unique_skills_raw": int(len(self.raw_frequency)),
            "unique_skills_canonical": int(len(self.canonical_frequency)),
            "min_frequency_threshold": int(min_frequency),
            "final_vocabulary_size": int(len(vocab)),
        }
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {
            "vocab_json": str(vocab_json_path),
            "vocab_csv": str(vocab_csv_path),
            "meta_json": str(meta_json_path),
        }

# Synonym map for common skill variants to canonical forms
DEFAULT_SYNONYM_MAP: Dict[str, str] = {
    "py": "python",
    "python 3": "python",
    "python 3.x": "python",

    "c sharp": "c#",
    "c plus plus": "c++",

    "js": "javascript",
    "nodejs": "node.js",
    "node js": "node.js",
    "node": "node.js",

    "postgres": "postgresql",
    "ms sql server": "sql server",
    "aws": "amazon web services",

    "ml": "machine learning",
    "powerbi": "power bi",

    "ai": "artificial intelligence",

    "problemsolving": "problem solving",
    "communicationskills": "communication skills",
    "timemanagement": "time management",
    "interpersonalskills": "interpersonal skills",
    "organizationalskills": "organizational skills",

    "rn": "registered nurse",
    "bls certification": "bls",
    "acls certification": "acls",
    "ons": "oncology nursing society",
    "csections": "c-sections",
    "fp ob": "fp/ob",
    "401 k": "401(k)",
    "401 k plan": "401(k)"
}