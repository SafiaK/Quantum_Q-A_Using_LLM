#!/usr/bin/env python3
"""
Script to select 2 examples from the filtered dataset and save to JSON
"""

import pandas as pd
import json
import random
from text_cleaning import clean_question_content, clean_answer_content, create_question_summary, create_answer_summary

def select_examples_from_data(filtered_file: str, output_file: str = "few_shot_examples.json", num_examples: int = 2):
    """
    Select examples from the filtered dataset and save to JSON
    """
    print(f"Loading filtered dataset from {filtered_file}...")
    df = pd.read_csv(filtered_file)
    
    print(f"Dataset shape: {df.shape}")
    
    # Filter for good examples (not too short, not too long)
    good_examples = df[
        (df['Answer Body'].str.len() >= 150) &
        (df['Answer Body'].str.len() <= 600) &
        (df['QuestionTitle'].str.len() >= 30) &
        (df['QuestionBody'].str.len() >= 80)
    ].copy()
    
    print(f"Found {len(good_examples)} good examples")
    
    if len(good_examples) < num_examples:
        print(f"Warning: Only {len(good_examples)} examples meet criteria")
        num_examples = len(good_examples)
    
    # Select diverse examples based on different question tags
    selected_examples = []
    selected_qids = set()
    
    # Try to select examples with different tags
    tag_groups = good_examples.groupby('QuestionTags')
    unique_tag_groups = list(tag_groups.groups.keys())
    
    # Shuffle to get random selection
    random.shuffle(unique_tag_groups)
    
    for tags in unique_tag_groups:
        if len(selected_examples) >= num_examples:
            break
            
        group = tag_groups.get_group(tags)
        if len(group) > 0:
            # Select a random example from this tag group
            selected_row = group.sample(1).iloc[0]
            
            # Clean the content
            cleaned_question = clean_question_content(
                selected_row['QuestionTitle'],
                selected_row['QuestionBody'],
                selected_row['QuestionTags']
            )
            
            cleaned_answer = clean_answer_content(selected_row['Answer Body'])
            
            example = {
                'question_id': int(selected_row['QuestionId']),
                'question_title': cleaned_question['title'],
                'question_body': cleaned_question['body'],
                'question_tags': cleaned_question['tags'],
                'answer_body': cleaned_answer,
                'question_summary': create_question_summary(cleaned_question),
                'answer_summary': create_answer_summary(cleaned_answer),
                'question_date': selected_row['QuestionDate'],
                'answer_date': selected_row['AnswerDate']
            }
            
            selected_examples.append(example)
            selected_qids.add(int(selected_row['QuestionId']))
    
    # If we don't have enough diverse examples, fill with random ones
    if len(selected_examples) < num_examples:
        remaining = num_examples - len(selected_examples)
        remaining_examples = good_examples[~good_examples['QuestionId'].isin(selected_qids)]
        
        if len(remaining_examples) > 0:
            sample_size = min(remaining, len(remaining_examples))
            sample_examples = remaining_examples.sample(sample_size)
            
            for _, row in sample_examples.iterrows():
                cleaned_question = clean_question_content(
                    row['QuestionTitle'],
                    row['QuestionBody'],
                    row['QuestionTags']
                )
                
                cleaned_answer = clean_answer_content(row['Answer Body'])
                
                example = {
                    'question_id': int(row['QuestionId']),
                    'question_title': cleaned_question['title'],
                    'question_body': cleaned_question['body'],
                    'question_tags': cleaned_question['tags'],
                    'answer_body': cleaned_answer,
                    'question_summary': create_question_summary(cleaned_question),
                    'answer_summary': create_answer_summary(cleaned_answer),
                    'question_date': row['QuestionDate'],
                    'answer_date': row['AnswerDate']
                }
                
                selected_examples.append(example)
                selected_qids.add(int(row['QuestionId']))
    
    # Save examples to JSON
    examples_data = {
        'examples': selected_examples,
        'selected_question_ids': list(selected_qids),
        'total_examples': len(selected_examples),
        'selection_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSelected {len(selected_examples)} examples:")
    for i, example in enumerate(selected_examples, 1):
        print(f"\nExample {i}:")
        print(f"Question ID: {example['question_id']}")
        print(f"Title: {example['question_title']}")
        print(f"Tags: {example['question_tags']}")
        print(f"Answer preview: {example['answer_summary'][:100]}...")
    
    print(f"\nExamples saved to {output_file}")
    print(f"Selected question IDs: {list(selected_qids)}")
    
    return examples_data

if __name__ == "__main__":
    # Select examples
    examples_data = select_examples_from_data("filtered_quantum_dataset.csv")

