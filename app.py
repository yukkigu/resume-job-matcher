import streamlit as st
import pdfplumber
import pandas as pd
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

def resume_to_df(text: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "ID": 0,
        "resume_cleaned": text
    }])

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
    model_weight = st.slider("Model weight", 0.0, 0.5, 0.4)
    dictionary_weight = st.slider("Dictionary weight", 0.0, 0.5, 0.4)

# =====================
# Main area
# =====================
st.title("PDF Parser")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
resume_text = ""
if uploaded_file:
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
        st.warning("Please upload a PDF file.")
        st.stop()
    
    with st.spinner("Computing recommendations..."):
        try:
            pre = PreprocessText(remove_stopwords=False)
            
            # --- resume cleaning ---
            clean_resume = pre.clean_text(resume_text)
            resume_df = resume_to_df(clean_resume)

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

            st.subheader("Top recommendations")
            st.dataframe(topk, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error during recommendation: {e}")
            st.stop()