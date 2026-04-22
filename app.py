import streamlit as st
import pdfplumber
import sys

from src.preprocessing import PreprocessText
from src.skills import SkillVocabularyBuilder, DEFAULT_SYNONYM_MAP

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Resume Job Matcher", layout="wide")
st.title("Resume Job Matcher")

# -------------------------
# Utils
# -------------------------
def parse_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"PDF parsing failed: {e}")
    return text


def load_text(file):
    if file.type == "application/pdf":
        return parse_pdf(file)
    else:
        return file.read().decode("utf-8", errors="ignore")


# -------------------------
# Init pipeline
# -------------------------
pre = PreprocessText(remove_stopwords=False)
extractor = SkillVocabularyBuilder(DEFAULT_SYNONYM_MAP)

# -------------------------
# Sidebar (job input)
# -------------------------
st.sidebar.header("Job Input")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)
job_input_mode = st.sidebar.radio(
    "Choose job input type",
    ["Paste Job Description", "Use Example"]
)

if job_input_mode == "Paste Job Description":
    job_text = st.sidebar.text_area(
        "Paste job description here",
        height=200, 
        width=800
    )
else:
    job_text = "Looking for a machine learning engineer with Python, AWS, and Docker experience."

# -------------------------
# Main input (resume)
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Resume (PDF or TXT)",
    type=["pdf", "txt"]
)

# -------------------------
# Run pipeline
# -------------------------
if uploaded_file:

    # ---------- Load ----------
    raw_text = load_text(uploaded_file)

    if not raw_text.strip():
        st.warning("No text extracted from file.")
        st.stop()

    # ---------- Preprocess ----------
    clean_text = pre.clean_text(raw_text)

    # ---------- Skill extraction ----------
    resume_skills = extractor.extract(clean_text)
    job_skills = extractor.extract(job_text)

    # ---------- Matching ----------
    resume_set = set()
    job_set = set()

    matched = resume_set & job_set
    missing = job_set - resume_set

    score = len(matched) / (len(job_set) + 1e-5)

    # -------------------------
    # UI Display
    # -------------------------

    # Score
    st.subheader("Match Score")
    st.progress(min(score, 1.0))
    st.write(f"{score:.2f}")

    # Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matched Skills")
        if matched:
            st.write(list(matched))
        else:
            st.write("None")

    with col2:
        st.subheader("Missing Skills")
        if missing:
            st.write(list(missing))
        else:
            st.write("None")

    # Recommendation
    st.subheader("Recommendations")
    if missing:
        st.write("You may consider learning:")
        st.write(list(missing))
    else:
        st.write("Good match. No major skill gaps.")

    # Debug section
    with st.expander("Debug Info"):
        st.subheader("Extracted Resume Skills")
        # st.write(resume_skills)

        st.subheader("Extracted Job Skills")
        # st.write(job_skills)

        st.subheader("Parsed Resume Text (first 1000 chars)")
        st.text(raw_text[:1000])