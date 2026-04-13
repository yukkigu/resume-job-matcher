# Resume–Job Matching and Skill Gap Analysis

## Overview

This project builds an NLP-based system that analyzes the fit between resumes and job descriptions. It extracts skills from unstructured resume text, compares them with job requirements, and provides an interpretable match score along with actionable recommendations.

## Dataset

### Resume Dataset

- Kaggle link: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- Contains more than 2400 resumes in string as well as PDF format. PDF stored in the data folder differentiated into their respective labels as folders with each resume inside the folder in pdf form with filename as the id defined in the csv

| Field       | Description                                                           |
| ----------- | --------------------------------------------------------------------- |
| ID          | Unique identifier and file name for the respective pdf                |
| Resume_str  | Contains the resume text only in string format                        |
| Resume_html | Contains the resume data in html format as present while web scraping |
| Category    | Category of the job the resume was used to apply                      |

- Jobs included in the dataset: HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation

### 1.3 M LinkedIn Jobs and Skills

- Kaggle Link: https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024
- Content: This dataset contains 1.3 million job listings scraped from LinkedIn in the year 2024.

| Field      | Description                                                                            |
| ---------- | -------------------------------------------------------------------------------------- |
| job_link   | 1296381 unique values of links to unique LinkedIn jobs                                 |
| job_skills | 1287105 unique values of corresponding list of skills for that particular LinkedIn job |

## Setup

- GUI: Streamlit

## Methodology

1. Text preprocessing (cleaning, normalization)
2. Skill extraction using dictionary-based and NLP methods
3. Skill normalization and synonym mapping
   Resume-job similarity computation
4. Explainable output generation (matched skills, missing skills, recommendations)

## Project Structure

```
project/
├── app.py
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── skill_extraction.ipynb
│   ├── matching.ipynb
│   └── final_demo.ipynb
├── src/
│   ├── preprocess.py
│   ├── skills.py
│   ├── matching.py
│   ├── recommend.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
```
