import pandas as pd
from dictionary_match import compute_dictionary_pairs_and_topk, compute_title_similarity
from embedder import compute_text_pairs_and_topk

def compute_fused_pairs_and_topk(
    resume_df: pd.DataFrame,
    job_df: pd.DataFrame,
    resume_skill_col: str = "extracted_skills_ranked",
    job_skill_col: str = "extracted_skills_ranked",
    resume_text_col: str = "resume_cleaned",
    job_desc_col: str = "job_summary",
    top_k: int = 5,
    model_name: str = "BAAI/bge-small-en-v1.5",
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
        0.4 * (fused["dictionary_score"] / 100.0)
        + 0.4 * fused["semantic_similarity"]
        + 0.2 * fused["title_similarity"]
    ) * 100.0

    fused = fused.sort_values(["resume_id", "final_score"], ascending=[True, False])
    topk = fused.groupby("resume_id", as_index=False).head(top_k).copy()
    topk["rank"] = topk.groupby("resume_id").cumcount() + 1

    return fused, topk