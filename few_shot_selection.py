#!/usr/bin/env python3
"""
Few-shot example selection for quantum Q&A dataset
"""

import pandas as pd
import random
from typing import List, Dict, Any, Tuple
from text_cleaning import clean_question_content, clean_answer_content, create_question_summary, create_answer_summary

def select_few_shot_examples(df: pd.DataFrame, num_examples: int = 2, 
                            min_answer_length: int = 100, max_answer_length: int = 500) -> List[Dict[str, Any]]:
    """
    Select few-shot examples from the filtered dataset
    """
    # Filter for good examples
    good_examples = df[
        (df['Answer Body'].str.len() >= min_answer_length) &
        (df['Answer Body'].str.len() <= max_answer_length) &
        (df['QuestionTitle'].str.len() >= 20) &
        (df['QuestionBody'].str.len() >= 50)
    ].copy()
    
    if len(good_examples) < num_examples:
        print(f"Warning: Only {len(good_examples)} examples meet criteria, using all available")
        num_examples = len(good_examples)
    
    # Randomly select examples
    selected_indices = random.sample(range(len(good_examples)), min(num_examples, len(good_examples)))
    selected_examples = good_examples.iloc[selected_indices]
    
    examples = []
    for _, row in selected_examples.iterrows():
        # Clean the content
        cleaned_question = clean_question_content(
            row['QuestionTitle'],
            row['QuestionBody'],
            row['QuestionTags']
        )
        
        cleaned_answer = clean_answer_content(row['Answer Body'])
        
        example = {
            'question_id': row['QuestionId'],
            'question_title': cleaned_question['title'],
            'question_body': cleaned_question['body'],
            'question_tags': cleaned_question['tags'],
            'answer_body': cleaned_answer,
            'question_summary': create_question_summary(cleaned_question),
            'answer_summary': create_answer_summary(cleaned_answer)
        }
        
        examples.append(example)
    
    return examples

def create_few_shot_prompt(examples: List[Dict[str, Any]], base_prompt: str, 
                          target_question: Dict[str, str]) -> str:
    """
    Create a few-shot prompt with examples and target question
    """
    prompt_parts = [base_prompt]
    
    # Add few-shot examples
    for i, example in enumerate(examples, 1):
        prompt_parts.append(f"\n### Example {i}:")
        prompt_parts.append(f"**Question:**")
        prompt_parts.append(f"Title: {example['question_title']}")
        if example['question_body']:
            prompt_parts.append(f"Body: {example['question_body']}")
        if example['question_tags']:
            prompt_parts.append(f"Tags: {example['question_tags']}")
        
        prompt_parts.append(f"\n**Answer:**")
        prompt_parts.append(example['answer_body'])
    
    # Add target question
    prompt_parts.append(f"\n### Target Question:")
    prompt_parts.append(f"**Question:**")
    prompt_parts.append(f"Title: {target_question['title']}")
    if target_question.get('body'):
        prompt_parts.append(f"Body: {target_question['body']}")
    if target_question.get('tags'):
        prompt_parts.append(f"Tags: {target_question['tags']}")
    
    prompt_parts.append(f"\n**Please provide your answer:**")
    
    return "\n".join(prompt_parts)

def select_diverse_examples(df: pd.DataFrame, num_examples: int = 2) -> List[Dict[str, Any]]:
    """
    Select diverse examples based on different criteria
    """
    # Group by question tags to ensure diversity
    tag_groups = df.groupby('QuestionTags')
    
    examples = []
    selected_tags = set()
    
    # Try to select examples from different tag groups
    for tags, group in tag_groups:
        if len(examples) >= num_examples:
            break
            
        # Skip if we've already selected from this tag group
        if tags in selected_tags:
            continue
            
        # Select a good example from this group
        good_examples = group[
            (group['Answer Body'].str.len() >= 100) &
            (group['Answer Body'].str.len() <= 500) &
            (group['QuestionTitle'].str.len() >= 20)
        ]
        
        if len(good_examples) > 0:
            selected_row = good_examples.sample(1).iloc[0]
            
            # Clean the content
            cleaned_question = clean_question_content(
                selected_row['QuestionTitle'],
                selected_row['QuestionBody'],
                selected_row['QuestionTags']
            )
            
            cleaned_answer = clean_answer_content(selected_row['Answer Body'])
            
            example = {
                'question_id': selected_row['QuestionId'],
                'question_title': cleaned_question['title'],
                'question_body': cleaned_question['body'],
                'question_tags': cleaned_question['tags'],
                'answer_body': cleaned_answer,
            
            }
            
            examples.append(example)
            selected_tags.add(tags)
    
    # If we don't have enough diverse examples, fill with random ones
    if len(examples) < num_examples:
        remaining = num_examples - len(examples)
        additional_examples = select_few_shot_examples(df, remaining)
        examples.extend(additional_examples)
    
    return examples[:num_examples]

def analyze_example_quality(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the quality of selected examples
    """
    if not examples:
        return {}
    
    analysis = {
        'total_examples': len(examples),
        'avg_question_length': 0,
        'avg_answer_length': 0,
        'unique_tags': set(),
        'difficulty_distribution': {'short': 0, 'medium': 0, 'long': 0}
    }
    
    total_q_len = 0
    total_a_len = 0
    
    for example in examples:
        q_len = len(example['question_title']) + len(example['question_body'])
        a_len = len(example['answer_body'])
        
        total_q_len += q_len
        total_a_len += a_len
        
        # Categorize by length
        if a_len < 200:
            analysis['difficulty_distribution']['short'] += 1
        elif a_len < 400:
            analysis['difficulty_distribution']['medium'] += 1
        else:
            analysis['difficulty_distribution']['long'] += 1
        
        # Extract tags
        if example['question_tags']:
            tags = [tag.strip() for tag in example['question_tags'].split(',')]
            analysis['unique_tags'].update(tags)
    
    analysis['avg_question_length'] = total_q_len / len(examples)
    analysis['avg_answer_length'] = total_a_len / len(examples)
    analysis['unique_tags'] = list(analysis['unique_tags'])
    
    return analysis

if __name__ == "__main__":
    # Test the few-shot selection
    print("Testing few-shot example selection...")
    
    # Load a sample of the dataset for testing
    try:
        df = pd.read_csv("filtered_quantum_dataset_with_valid_answers.csv")
        print(f"Loaded dataset with {len(df)} rows")
        
        # Select examples
        examples = select_diverse_examples(df, 2)
        print(f"Selected {len(examples)} examples")
        
        # Analyze quality
        analysis = analyze_example_quality(examples)
        print("\nExample Quality Analysis:")
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        # Show selected examples
        print("\nSelected Examples:")
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print(f"Question: {example['question_title']}")
            print(f"Tags: {example['question_tags']}")
            
    except FileNotFoundError:
        print("Filtered dataset not found. Please run data_filtering.py first.")
