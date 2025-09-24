#!/usr/bin/env python3
"""
Data filtering script to extract quantum Q&A data without href references
"""

import pandas as pd
import re
import json
from typing import List, Dict, Any

def has_href_references(text: str) -> bool:
    """
    Check if text contains any href references or URLs
    """
    if not isinstance(text, str):
        return False
    
    # Patterns to detect href references and URLs
    href_patterns = [
        r'href\s*=',  # href= attribute
        r'http[s]?://',  # http/https URLs
        r'www\.',  # www. URLs
        r'<a\s+[^>]*href',  # <a href> tags
        r'\[.*?\]\(.*?\)',  # markdown links
        r'https?://[^\s<>"]+',  # full URLs
        r'www\.[^\s<>"]+',  # www URLs
    ]
    
    for pattern in href_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

def clean_html_tags(text: str) -> str:
    """
    Clean HTML tags from text while preserving content
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags but keep content
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def filter_dataset(input_file: str, output_file: str) -> Dict[str, Any]:
    """
    Filter dataset to remove rows with href references
    """
    print(f"Loading dataset from {input_file}...")
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Could not load CSV file with any supported encoding")
    
    print(f"Original dataset shape: {df.shape}")
    
    # Filter out rows where QuestionBody or Answer Body contain href references
    print("Filtering out rows with href references...")
    
    # Create boolean masks for filtering
    question_has_href = df['QuestionBody'].apply(has_href_references)
    answer_has_href = df['Answer Body'].apply(has_href_references)
    
    # Keep rows where neither question nor answer has href references
    clean_mask = ~(question_has_href | answer_has_href)
    filtered_df = df[clean_mask].copy()
    
    print(f"Filtered dataset shape: {filtered_df.shape}")
    print(f"Removed {df.shape[0] - filtered_df.shape[0]} rows with href references")
    
    # Clean the text content
    print("Cleaning text content...")
    filtered_df['QuestionBody'] = filtered_df['QuestionBody'].apply(clean_html_tags)
    filtered_df['Answer Body'] = filtered_df['Answer Body'].apply(clean_html_tags)
    
    # Select required columns
    required_columns = [
        'QuestionId', 'QuestionTitle', 'QuestionBody', 'QuestionTags', 
        'QuestionDate', 'AcceptedAnswerId', 'AnswerId', 'Answer Body', 'AnswerDate'
    ]
    
    # Check which columns exist
    available_columns = [col for col in required_columns if col in filtered_df.columns]
    final_df = filtered_df[available_columns].copy()
    
    # Save filtered dataset
    final_df.to_csv(output_file, index=False)
    print(f"Saved filtered dataset to {output_file}")
    
    # Calculate statistics
    stats = {
        'original_rows': df.shape[0],
        'filtered_rows': final_df.shape[0],
        'removed_rows': df.shape[0] - final_df.shape[0],
        'questions_with_multiple_answers': 0,
        'unique_questions': 0
    }
    
    # Count questions with multiple answers
    if 'QuestionId' in final_df.columns:
        question_answer_counts = final_df.groupby('QuestionId').size()
        stats['questions_with_multiple_answers'] = (question_answer_counts > 1).sum()
        stats['unique_questions'] = len(question_answer_counts)
    
    return stats

def filter_dataset_with_tags(input_file: str, output_file: str) -> Dict[str, Any]:
    """
    Filter dataset to remove rows with href references but keep tags intact
    """
    print(f"Loading dataset from {input_file}...")
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Could not load CSV file with any supported encoding")
    
    print(f"Original dataset shape: {df.shape}")
    
    # Filter out rows where QuestionBody or Answer Body contain href references
    print("Filtering out rows with href references...")
    
    # Create boolean masks for filtering
    question_has_href = df['QuestionBody'].apply(has_href_references)
    answer_has_href = df['Answer Body'].apply(has_href_references)
    
    # Keep rows where neither question nor answer has href references
    clean_mask = ~(question_has_href | answer_has_href)
    filtered_df = df[clean_mask].copy()
    
    print(f"Filtered dataset shape: {filtered_df.shape}")
    print(f"Removed {df.shape[0] - filtered_df.shape[0]} rows with href references")
    
    # Clean the text content but preserve tags
    print("Cleaning text content (preserving tags)...")
    filtered_df['QuestionBody'] = filtered_df['QuestionBody'].apply(clean_html_tags)
    filtered_df['Answer Body'] = filtered_df['Answer Body'].apply(clean_html_tags)
    # Keep QuestionTags as is - don't clean them
    
    # Select required columns
    required_columns = [
        'QuestionId', 'QuestionTitle', 'QuestionBody', 'QuestionTags', 
        'QuestionDate', 'AcceptedAnswerId', 'AnswerId', 'Answer Body', 'AnswerDate'
    ]
    
    # Check which columns exist
    available_columns = [col for col in required_columns if col in filtered_df.columns]
    final_df = filtered_df[available_columns].copy()
    
    # Save filtered dataset
    final_df.to_csv(output_file, index=False)
    print(f"Saved filtered dataset with tags to {output_file}")
    
    # Calculate statistics
    stats = {
        'original_rows': df.shape[0],
        'filtered_rows': final_df.shape[0],
        'removed_rows': df.shape[0] - final_df.shape[0],
        'questions_with_multiple_answers': 0,
        'unique_questions': 0
    }
    
    # Count questions with multiple answers
    if 'QuestionId' in final_df.columns:
        question_answer_counts = final_df.groupby('QuestionId').size()
        stats['questions_with_multiple_answers'] = (question_answer_counts > 1).sum()
        stats['unique_questions'] = len(question_answer_counts)
    
    return stats

def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """
    Analyze the filtered dataset for statistics
    """
    df = pd.read_csv(file_path)
    
    stats = {
        'total_rows': len(df),
        'unique_questions': df['QuestionId'].nunique() if 'QuestionId' in df.columns else 0,
        'questions_with_multiple_answers': 0,
        'avg_answers_per_question': 0,
        'date_range': {}
    }
    
    if 'QuestionId' in df.columns:
        question_answer_counts = df.groupby('QuestionId').size()
        stats['questions_with_multiple_answers'] = (question_answer_counts > 1).sum()
        stats['avg_answers_per_question'] = question_answer_counts.mean()
    
    if 'QuestionDate' in df.columns:
        df['QuestionDate'] = pd.to_datetime(df['QuestionDate'], errors='coerce')
        stats['date_range'] = {
            'earliest': df['QuestionDate'].min(),
            'latest': df['QuestionDate'].max()
        }
    
    return stats

if __name__ == "__main__":
    input_file = "Quantum_Dataset 26-06-2024.csv"
    output_file = "filtered_quantum_dataset.csv"
    
    # Filter the dataset
    stats = filter_dataset(input_file, output_file)
    
    print("\n" + "="*50)
    print("FILTERING STATISTICS")
    print("="*50)
    print(f"Original rows: {stats['original_rows']}")
    print(f"Filtered rows: {stats['filtered_rows']}")
    print(f"Removed rows: {stats['removed_rows']}")
    print(f"Questions with multiple answers: {stats['questions_with_multiple_answers']}")
    print(f"Unique questions: {stats['unique_questions']}")
    
    # Analyze the filtered dataset
    print("\n" + "="*50)
    print("FILTERED DATASET ANALYSIS")
    print("="*50)
    analysis = analyze_dataset(output_file)
    for key, value in analysis.items():
        print(f"{key}: {value}")
