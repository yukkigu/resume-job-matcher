# Utility functions for data loading, preprocessing, and helpers.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

# Load resume data
def load_resume_data(data_dir: str) -> pd.DataFrame:
    """Load resume dataset from CSV."""
    csv_path = os.path.join(data_dir, 'raw', 'resume-data', 'Resume', 'Resume.csv')
    df = pd.read_csv(csv_path)
    return df

# Load job description data
def load_job_skills(data_dir: str) -> pd.DataFrame:
    """Load job skills dataset from CSV."""
    csv_path = os.path.join(data_dir, 'raw', 'job-description-data', 'job_skills.csv')
    df = pd.read_csv(csv_path)
    return df

# Load job summary data
def load_job_summary(data_dir: str) -> pd.DataFrame:
    """Load job summary dataset from CSV."""
    csv_path = os.path.join(data_dir, 'raw', 'job-description-data', 'job_summary.csv')
    df = pd.read_csv(csv_path)
    return df

# Load LinkedIn job postings data
def load_linkedin_jobs(data_dir: str) -> pd.DataFrame:
    """Load LinkedIn jobs dataset from CSV."""
    csv_path = os.path.join(data_dir, 'raw', 'job-description-data', 'linkedin_job_postings.csv')
    df = pd.read_csv(csv_path)
    return df

# Create processed data directory if it doesn't exist
def ensure_processed_dir(data_dir: str) -> str:
    """Create processed data directory if it doesn't exist."""
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir

# Save and load JSON utilities
def save_json(data: Union[Dict, List], filepath: str) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: str) -> Union[Dict, List]:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Sampling utilities (stratified by category for resumes, random for jobs)

# Ensures we have around the same number of samples per category, 
# If a category has fewer than n_per_category, it takes all of them
def sample_resumes_stratified(df: pd.DataFrame, n_per_category: int = 20) -> pd.DataFrame:
    """
    Sample resumes stratified by category.

    Args:
        df: Resume dataframe with 'Category' column
        n_per_category: Number of resumes to sample per category

    Returns:
        Sampled dataframe
    """
    if 'Category' not in df.columns:
        return df.sample(min(len(df), n_per_category * 10), random_state=42).reset_index(drop=True)

    # Robust against pandas groupby.apply behavior changes
    sampled_parts = []
    for _, group in df.groupby('Category', dropna=False):
        n_take = min(len(group), n_per_category)
        sampled_parts.append(group.sample(n=n_take, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled

# Ensures we have a random sample of jobs, 
# If the dataset is smaller than n_samples, it takes all of them
def sample_jobs_stratified(df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    """
    Sample jobs randomly (stratified by availability).
    
    Args:
        df: Job dataframe
        n_samples: Number of jobs to sample
    
    Returns:
        Sampled dataframe
    """
    sample_size = min(len(df), n_samples)
    return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

# String utilities

# Cleans text
def clean_text(text: str) -> str:
    """
    Basic text cleaning (remove special chars, normalize whitespace).
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # remove extra whitespace
    text = ' '.join(text.split())
    # remove special characters but keep alphanumeric, spaces, and common punctuation
    text = ''.join(c if c.isalnum() or c in ' .,;:' else ' ' for c in text)
    return text.lower().strip()

# Safe nested dictionary access
def safe_get(obj: Dict, *keys: str, default: str = "") -> str:
    """Safe nested dictionary access."""
    # traverse the keys in the dictionary
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        else:
            return default
    return str(obj) if obj is not None else default