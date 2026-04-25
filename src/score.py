import pandas as pd
from src.dictionary_match import compute_dictionary_pairs_and_topk, compute_title_similarity
from src.embedder import compute_text_pairs_and_topk

def compute_fused_pairs_and_topk(
    resume_df: pd.DataFrame,
    job_df: pd.DataFrame,
    resume_skill_col: str = "extracted_skills_ranked",
    job_skill_col: str = "extracted_skills_ranked",
    resume_text_col: str = "resume_cleaned",
    job_desc_col: str = "job_summary",
    top_k: int = 5,
    model_name: str = "BAAI/bge-small-en-v1.5",
    skill_weight: float = 0.4,
    model_weight: float = 0.4,
    title_weight: float = 0.2,
):
    dict_pairs, _ = compute_dictionary_pairs_and_topk(
        resume_df=resume_df,
        job_df=job_df,
        resume_skill_col=resume_skill_col,
        job_skill_col=job_skill_col,
        top_k=top_k,
    )

    sem_pairs, _ = compute_text_pairs_and_topk(
        resume_df=resume_df,
        job_df=job_df,
        resume_text_col=resume_text_col,
        job_desc_col=job_desc_col,
        top_k=top_k,
        model_name=model_name,
    )

    fused = dict_pairs.merge(
        sem_pairs[["resume_id", "job_idx", "semantic_similarity"]],
        on=["resume_id", "job_idx"],
        how="inner",
        suffixes=("_dict", "_sem"),
    )

    fused["title_similarity"] = fused.apply(
        lambda row: compute_title_similarity(row["resume_category"], row["job_title"]),
        axis=1,
    )

    fused["final_score"] = (
        skill_weight * (fused["dictionary_score"] / 100.0)
        + model_weight * fused["semantic_similarity"]
        + title_weight * fused["title_similarity"]
    ) * 100.0

    fused = fused.sort_values(["resume_id", "final_score"], ascending=[True, False])
    topk = fused.groupby("resume_id", as_index=False).head(top_k).copy()
    topk["rank"] = topk.groupby("resume_id").cumcount() + 1

    return fused, topk

def save_model_scores(df, output_path):
    required_cols = ["resume_id", "job_idx"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

def compute_final_scores(
    skill_pairs_df: pd.DataFrame,
    model_pairs_df: pd.DataFrame,
    title_pairs_df: pd.DataFrame,
    skill_weight: float = 0.4,
    model_weight: float = 0.4,
    title_weight: float = 0.2,
    output_path: str = "../data/processed/final_scores.csv",
):
    if not (0 <= skill_weight <= 1 and 0 <= model_weight <= 1 and 0 <= title_weight <= 1):
        raise ValueError("Weights must be between 0 and 1")
    if round(skill_weight + model_weight + title_weight, 5) != 1.0:
        raise ValueError("Weights must sum to 1")

    merged = skill_pairs_df.merge(
        model_pairs_df[["resume_id", "job_idx", "semantic_similarity", "model_name"]],
        on=["resume_id", "job_idx"],
        how="inner",
        suffixes=("_skill", "_model"),
    ).merge(
        title_pairs_df[["resume_id", "job_idx", "title_similarity"]],
        on=["resume_id", "job_idx"],
        how="inner",
    )

    merged["final_score"] = (
        skill_weight * (merged["dictionary_score"] / 100.0)
        + model_weight * merged["semantic_similarity"]
        + title_weight * merged["title_similarity"]
    ) * 100.0

    merged.to_csv(output_path, index=False)
    print(f"Final scores saved to {output_path}")
    return merged

def compute_ranks_and_topk(
    final_scores_df: pd.DataFrame,
    score_col: str = "final_score",
    resume_id_col: str = "resume_id",
    top_k: int = 5,
    output_ranked_path: str | None = "../data/scores/final_ranked.csv",
    output_topk_path: str | None = "../data/scores/final_topk.csv",
):
    """
    Compute per-resume ranks and Top-K rows.

    Args:
        final_scores_df: DataFrame containing at least resume_id_col and score_col
        score_col: column to rank by
        resume_id_col: resume identifier column
        top_k: number of top rows to keep per resume
        output_ranked_path: optional CSV path to save full ranked dataframe
        output_topk_path: optional CSV path to save top-k dataframe

    Returns:
        ranked_df, topk_df
    """
    required_cols = [resume_id_col, score_col]
    for col in required_cols:
        if col not in final_scores_df.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column")

    # optional stable sort if job_idx exists
    sort_cols = [resume_id_col, score_col]
    ascending = [True, False]

    if "job_idx" in final_scores_df.columns:
        sort_cols.append("job_idx")
        ascending.append(True)

    ranked_df = final_scores_df.sort_values(
        sort_cols,
        ascending=ascending
    ).copy()

    ranked_df["rank"] = ranked_df.groupby(resume_id_col).cumcount() + 1

    topk_df = ranked_df[ranked_df["rank"] <= top_k].copy()

    if output_ranked_path is not None:
        ranked_df.to_csv(output_ranked_path, index=False)
        print(f"Full ranked results saved to {output_ranked_path}")

    if output_topk_path is not None:
        topk_df.to_csv(output_topk_path, index=False)
        print(f"Top-{top_k} results saved to {output_topk_path}")

    return ranked_df, topk_df

def get_topk_for_resume(
    ranked_df: pd.DataFrame,
    resume_id,
    k: int = 5,
    resume_id_col: str = "resume_id",
) -> pd.DataFrame:
    if resume_id_col not in ranked_df.columns or "rank" not in ranked_df.columns:
        raise ValueError("DataFrame must contain resume_id column and rank column")

    return ranked_df[
        (ranked_df[resume_id_col] == resume_id) &
        (ranked_df["rank"] <= k)
    ].copy()