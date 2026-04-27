import streamlit as st
import pdfplumber
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from src.score import compute_fused_pairs_and_topk, compute_final_scores, compute_ranks_and_topk
from src.preprocessing import PreprocessText
from src.skills_extraction import SkillExtractor
from src.skills import DEFAULT_SYNONYM_MAP

def parse_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"PDF parsing failed: {e}")
    return text

def resume_to_df(text: str, resume_title: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "ID": 0,
        "resume_cleaned": text,
        # score.compute_fused_pairs_and_topk uses this for title similarity
        "Category": resume_title or ""
    }])

def _skill_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    # try JSON list string
    try:
        import json

        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass
    # fallback: comma-separated
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]

def build_improvement_report(topk: pd.DataFrame, resume_title: str, weights: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Top-5 Job Recommendations: Improvement Report")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Resume title/target role: {resume_title or '(not provided)'}")
    lines.append(
        f"- Weights: dictionary={weights['dictionary_weight']:.2f}, "
        f"semantic={weights['model_weight']:.2f}, title={weights['title_weight']:.2f}"
    )
    lines.append("")

    any_missing = False
    for _, row in topk.iterrows():
        rank = int(row.get("rank", 0))
        job_title = str(row.get("job_title", ""))
        company = str(row.get("company", ""))
        final_score = row.get("final_score", "")
        dict_score = row.get("dictionary_score", "")
        semantic = row.get("semantic_similarity", "")
        title_sim = row.get("title_similarity", "")

        missing_skills = _skill_list(row.get("missing_skills", []))
        missing_top = missing_skills[:12]
        if missing_skills:
            any_missing = True

        lines.append(f"## Rank {rank}: {job_title} — {company}")
        lines.append("")
        lines.append(
            f"- Final score: {final_score}\n"
            f"- Dictionary score: {dict_score}\n"
            f"- Semantic score: {semantic}\n"
            f"- Title similarity: {title_sim}"
        )
        lines.append("")
        lines.append("**Missing skills to focus on**")
        lines.append("")
        if missing_top:
            for s in missing_top:
                lines.append(f"- {s}")
        else:
            lines.append("- (none detected)")
        lines.append("")
        lines.append("**Advice to improve fit**")
        lines.append("")
        if not missing_skills:
            lines.append("- Tailor your resume bullets to mirror the job description’s keywords.")
            lines.append("- Quantify impact (numbers, scope, time saved, revenue, etc.).")
            lines.append("- Add a short “Relevant Projects” section if applicable.")
        else:
            lines.append(f"- Prioritize learning or demonstrating: {', '.join(missing_top[:6])}.")
        lines.append("")

    if any_missing:
        lines.append("## General next steps (apply to all roles)")
        lines.append("")
        lines.append("- Add 1–2 resume bullets per missing skill showing evidence (project/work/academic).")
        lines.append("- Build one small portfolio project using 2–3 of the missing skills together.")
        lines.append("- For interviews: prepare a STAR story for each top missing skill.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"

# =====================
# Page config
# =====================
st.set_page_config(page_title="Resume → Job Recommender", layout="wide")
st.title("Resume → Job Recommender")

# =====================
# Sidebar
# =====================
with st.sidebar:
    st.header("Settings")

    top_k = st.slider("Top K jobs", 1, 20, 5)

    method = st.selectbox(
        "Model",
        [
            "BGE-small (composite scoring)", 
            "MiniLM (composite scoring)", 
            "MPNet (composite scoring)",
            "Skills + TF-IDF (hybrid)"
        ]
    )

    if method == "BGE-small (composite scoring)":
        method = "BAAI/bge-small-en-v1.5"
    elif method == "MiniLM (composite scoring)":
        method = "sentence-transformers/all-MiniLM-L6-v2"
    elif method == "MPNet (composite scoring)":
        method = "sentence-transformers/all-mpnet-base-v2"
    elif method == "Skills + TF-IDF (hybrid)":
        method = "tfidf"

    title_weight = st.slider("Title weight", 0.0, 0.5, 0.2)
    model_weight = st.slider("Model weight", 0.0, 0.5, 0.5)
    dictionary_weight = st.slider("Dictionary weight", 0.0, 0.5, 0.3)

# =====================
# Main area
# =====================
st.subheader("Resume input")

resume_title = st.text_input("Resume title / target role (used for title similarity)", value="")

input_mode = st.radio("Resume input type", ["Paste text", "Upload PDF"], horizontal=True)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
resume_text = ""
if input_mode == "Paste text":
    resume_text = st.text_area("Paste resume text", height=250, placeholder="Paste resume text here…")
elif uploaded_file:
    resume_text = parse_pdf(uploaded_file)

    if not resume_text.strip():
        st.warning("No text extracted.")

# =====================
# Button
# =====================
col1, col2 = st.columns([1, 4])

with col1:
    run = st.button("Get Recommendations", type="primary")

# =====================
# Output area
# =====================
if run:
    if not resume_text.strip():
        st.warning("Please provide resume text (paste or upload PDF).")
        st.stop()
    
    with st.spinner("Computing recommendations..."):
        try:
            pre = PreprocessText(remove_stopwords=False)
            
            # --- resume cleaning ---
            clean_resume = pre.clean_text(resume_text)
            resume_df = resume_to_df(clean_resume, resume_title=resume_title)

            # --- skill extraction for resume ---
            skill_extractor = SkillExtractor.from_vocabulary_json(
                vocab_path="./data/processed/skill_vocabulary.json",
                synonym_map=DEFAULT_SYNONYM_MAP,
                min_alias_len=2,
                min_vocab_frequency=10,
            )

            resume_out = skill_extractor.extract_from_dataframe(
                resume_df,
                text_col="resume_cleaned",
                output_col="extracted_skills_ranked",
                max_skills_per_text=300,
            )

            # --- load job data ---
            processed_dir = "./data/processed"
            job_df = f"{processed_dir}/job_skills_extracted.csv"
            job_out = pd.read_csv(job_df)


            if model_weight > 1 or title_weight > 1 or dictionary_weight > 1 or model_weight < 0 or title_weight < 0 or dictionary_weight < 0:
                st.error("Weights must be between 0 and 1.")
                st.stop()
            if round(model_weight + title_weight + dictionary_weight, 5) != 1.0:
                st.error("Weights must sum to 1.")
                st.stop()

            # --- compute scores and top-k ---
            pairs, topk = compute_fused_pairs_and_topk(
                resume_df=resume_out,
                job_df=job_out,
                top_k=top_k,
                model_name=method,
                skill_weight=dictionary_weight,
                model_weight=model_weight,
                title_weight=title_weight,
            )

            topk.drop(columns=["resume_id", "resume_idx", "resume_category"], inplace=True, errors="ignore")
            st.subheader("Top recommendations (original output)")
            st.dataframe(topk, use_container_width=True, hide_index=True)

            # ---- Detailed scoring table ----
            st.subheader("Top recommendations (detailed scoring)")
            detailed = topk.copy()
            # keep only the most relevant columns if present
            cols = [
                "rank",
                "job_title",
                "company",
                "final_score",
                "dictionary_score",
                "semantic_similarity",
                "title_similarity",
                "matched_skills",
                "missing_skills",
            ]
            cols = [c for c in cols if c in detailed.columns]
            detailed = detailed[cols]
            detailed.insert(
                3,
                "weights_used",
                f"dict={dictionary_weight:.2f}, semantic={model_weight:.2f}, title={title_weight:.2f}",
            )
            st.dataframe(detailed, use_container_width=True, hide_index=True)

            # ---- Downloadable advice report ----
            st.subheader("Downloadable improvement report")
            weights = {
                "dictionary_weight": dictionary_weight,
                "model_weight": model_weight,
                "title_weight": title_weight,
            }
            report_md = build_improvement_report(topk, resume_title=resume_title, weights=weights)
            out_dir = Path("./outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "top5_improvement_report.md"
            report_path.write_text(report_md, encoding="utf-8")

            st.download_button(
                "Download report (Markdown)",
                data=report_md.encode("utf-8"),
                file_name=report_path.name,
                mime="text/markdown",
            )

            # Optional CSV (topk with list columns JSON-serialized)
            csv_df = topk.copy()
            for col in ("matched_skills", "missing_skills", "extra_skills"):
                if col in csv_df.columns:
                    csv_df[col] = csv_df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download top-5 table (CSV)",
                data=csv_bytes,
                file_name="top5_recommendations.csv",
                mime="text/csv",
            )

            with st.expander("Report preview"):
                st.markdown(report_md)

        except Exception as e:
            st.error(f"Error during recommendation: {e}")
            st.stop()
