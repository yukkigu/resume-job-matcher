"""Skill extraction module

This module implements a dictionary-based skill extractor that uses a canonical skill vocabulary
with support for synonyms and alias matching.
"""

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.skills import SkillVocabularyBuilder, DEFAULT_SYNONYM_MAP


class SkillExtractor:
    """Dictionary-based skill extractor using canonical vocabulary + alias matching."""

    NOISE_SKILLS = {
        "a", "an", "the", "be", "to", "of", "in", "on", "at", "by", "or", "and",
        "work", "working", "skills", "skill", "experience", "knowledge", "ability",
        "abilities", "general", "review", "process", "processes", "office", "support",
        "reports", "reporting", "documentation", "education", "professional", "environment",
        "service", "services", "summary", "company", "name", "company name", "city", "state", "current",
        "activities", "issues", "managed", "management", "develop", "development",
        "access", "attitude", "care", "electronic", "acquisition", "compensation",
        "assignment", "assignments", "eligible", "consultant", "accomplishments",
        "knowledge of", "over", "time", "strong",
    }

    BENEFIT_PATTERNS = [
        r"\b401k\b",
        r"\b401\s*k\b",
        r"\b401\(k\)\b",
        r"\b401\s*k\s+savings\b",
        r"\bsavings plan\b",
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

    NON_SKILL_PHRASE_PATTERNS = [
        r"\b\d+\+?\s*(?:years?|yrs?)\b",
        r"\b(?:years?|yrs?)\s+(?:of\s+)?experience\b",
        r"\b\d+\s*(?:hour|hr)\b",
        r"\b(?:high school|bachelor(?:s|'s)?|master(?:s|'s)?|phd|doctorate)\b",
        r"\bcompany name\b",
    ]

    def __init__(
        self,
        vocabulary: Dict[str, int],
        synonym_map: Optional[Dict[str, str]] = None,
        min_alias_len: int = 2,
        min_vocab_frequency: int = 1,
    ):
        self.skill_builder = SkillVocabularyBuilder(synonym_map or DEFAULT_SYNONYM_MAP)
        self.synonym_map = self.skill_builder.synonym_map
        self.min_alias_len = min_alias_len
        self.min_vocab_frequency = min_vocab_frequency

        # canonical vocabulary normalized + sanitized
        self.skill_vocab = {
            self.skill_builder.canonicalize(self.skill_builder.normalize_skill(k)): int(v)
            for k, v in vocabulary.items()
            if self.skill_builder.normalize_skill(k)
            and int(v) >= self.min_vocab_frequency
            and self._is_meaningful_skill(self.skill_builder.canonicalize(self.skill_builder.normalize_skill(k)))
        }

        # alias -> canonical
        self.alias_to_canonical: Dict[str, str] = {}
        for canonical in self.skill_vocab.keys():
            self.alias_to_canonical[canonical] = canonical

        # include synonyms only when canonical exists in vocabulary
        for alias, canonical in self.synonym_map.items():
            alias_n = self.skill_builder.normalize_skill(alias)
            canonical_n = self.skill_builder.canonicalize(self.skill_builder.normalize_skill(canonical))
            if (
                alias_n
                and len(alias_n) >= self.min_alias_len
                and canonical_n in self.skill_vocab
                and self._is_meaningful_skill(alias_n)
            ):
                self.alias_to_canonical[alias_n] = canonical_n

        # Build a token n-gram index for fast matching.
        # This avoids running a regex search for every alias on every text.
        self._aliases_by_token_len: Dict[int, Dict[Tuple[str, ...], str]] = {}
        for alias, canonical in self.alias_to_canonical.items():
            alias_tokens = tuple(alias.split())
            if not alias_tokens:
                continue
            token_len = len(alias_tokens)
            if token_len not in self._aliases_by_token_len:
                self._aliases_by_token_len[token_len] = {}
            self._aliases_by_token_len[token_len][alias_tokens] = canonical

        self._token_lengths_desc = sorted(self._aliases_by_token_len.keys(), reverse=True)

    def _is_meaningful_skill(self, skill: str) -> bool:
        skill = self.skill_builder.normalize_skill(skill)
        if not skill:
            return False

        if skill in self.NOISE_SKILLS:
            return False

        if not self.skill_builder.is_valid_skill_phrase(skill):
            return False

        if re.fullmatch(r"\d+(?:\.\d+)?", skill):
            return False

        if re.fullmatch(r"(?:19|20)\d{2}", skill):
            return False

        for pattern in self.BENEFIT_PATTERNS:
            if re.search(pattern, skill):
                return False

        for pattern in self.NON_SKILL_PHRASE_PATTERNS:
            if re.search(pattern, skill):
                return False

        # reject very short generic single words unless known technical shorthand
        allow_short = {"c", "c#", "c++", "r", "go", "sql", "aws", "gcp", "api", "etl", "bi"}
        if skill not in allow_short and len(skill.split()) == 1 and len(skill) <= 3:
            return False

        return True

    @staticmethod
    def load_vocabulary_json(vocab_path: str) -> Dict[str, int]:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict):
            raise ValueError("Vocabulary JSON must be a dictionary: {skill: count}")
        return {str(k): int(v) for k, v in vocab.items()}

    @classmethod
    def from_vocabulary_json(
        cls,
        vocab_path: str,
        synonym_map: Optional[Dict[str, str]] = None,
        min_alias_len: int = 2,
        min_vocab_frequency: int = 1,
    ) -> "SkillExtractor":
        vocab = cls.load_vocabulary_json(vocab_path)
        return cls(
            vocabulary=vocab,
            synonym_map=synonym_map,
            min_alias_len=min_alias_len,
            min_vocab_frequency=min_vocab_frequency,
        )

    def extract_from_text(self, text: Any, max_skills: Optional[int] = None) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        normalized_text = self.skill_builder.normalize_skill(text)
        tokens = normalized_text.split()
        if not tokens:
            return []

        found = set()
        token_count = len(tokens)

        for n in self._token_lengths_desc:
            if n > token_count:
                continue

            lookup = self._aliases_by_token_len[n]
            for i in range(token_count - n + 1):
                phrase = tuple(tokens[i : i + n])
                canonical = lookup.get(phrase)
                if canonical is None:
                    continue
                found.add(canonical)
                if max_skills is not None and len(found) >= max_skills:
                    return sorted(found)

        return sorted(found)

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        output_col: str = "extracted_skills",
        max_skills_per_text: Optional[int] = None,
    ) -> pd.DataFrame:
        if text_col not in df.columns:
            raise KeyError(f"Column '{text_col}' not found in dataframe")

        out = df.copy()
        out[output_col] = out[text_col].apply(
            lambda x: self.extract_from_text(x, max_skills=max_skills_per_text)
        )
        out["extracted_skill_count"] = out[output_col].apply(len)
        return out


def compute_extraction_stats(
    df: pd.DataFrame,
    skills_col: str = "extracted_skills",
    top_n: int = 25,
) -> Dict[str, Any]:
    if skills_col not in df.columns:
        raise KeyError(f"Column '{skills_col}' not found in dataframe")

    total_rows = len(df)
    with_skills = int(df[skills_col].apply(lambda x: isinstance(x, list) and len(x) > 0).sum())
    avg_count = float(df[skills_col].apply(lambda x: len(x) if isinstance(x, list) else 0).mean())

    counter = Counter()
    for skills in df[skills_col]:
        if isinstance(skills, list):
            counter.update(skills)

    return {
        "total_rows": int(total_rows),
        "rows_with_skills": int(with_skills),
        "coverage_pct": round((with_skills / total_rows * 100.0), 2) if total_rows else 0.0,
        "avg_skills_per_row": round(avg_count, 2),
        "top_skills": counter.most_common(top_n),
    }