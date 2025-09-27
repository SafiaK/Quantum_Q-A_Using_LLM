#!/usr/bin/env python3
"""
Modified pipeline to run on first 3 rows using saved examples
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# Import our custom modules
from text_cleaning import clean_question_content, clean_answer_content
from ollama_integration import OllamaDeepSeek, load_prompt_template
from json_extraction import parse_llm_response

def load_examples_from_json(json_file: str = "few_shot_examples.json") -> List[Dict[str, Any]]:
    """
    Load few-shot examples from JSON file
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['examples']
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please run select_examples.py first.")
        return []
    except Exception as e:
        print(f"Error loading examples: {e}")
        return []

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

def run_pipeline_on_first_n_rows(filtered_file: str, examples_file: str, 
                                num_rows: int = 3, start_row: int = 0,
                                prompt1_file: str = "prompt1.txt") -> List[Dict[str, Any]]:
    """
    Run pipeline on first N rows of the filtered dataset
    """
    print(f"Running pipeline on rows {start_row} to {start_row + num_rows - 1}")
    print("="*60)
    
    # Load examples
    print("Loading few-shot examples...")
    examples = load_examples_from_json(examples_file)
    if not examples:
        raise ValueError("Could not load examples")
    
    example_qids = [ex['question_id'] for ex in examples]
    print(f"Loaded {len(examples)} examples with QIDs: {example_qids}")
    
    # Load filtered dataset
    print("Loading filtered dataset...")
    df = pd.read_csv(filtered_file)
    print(f"Dataset shape: {df.shape}")
    
    # Load prompt template
    print("Loading prompt template...")
    prompt1 = load_prompt_template(prompt1_file)
    
    if not prompt1:
        raise ValueError("Could not load prompt template")
    
    # Initialize Ollama client
    print("Initializing Ollama client...")
    client = OllamaDeepSeek()
    
    # Test connection
    test_result = client.test_connection()
    if not test_result['ollama_running'] or not test_result['model_available']:
        raise RuntimeError(f"Ollama not available: {test_result}")
    
    # Select rows to process (excluding example rows)
    print(f"Selecting rows {start_row} to {start_row + num_rows - 1} (excluding example rows)...")
    
    # Filter out example rows
    available_rows = df[~df['QuestionId'].isin(example_qids)].copy()
    print(f"Available rows after excluding examples: {len(available_rows)}")
    
    # Select the specified range
    end_row = min(start_row + num_rows, len(available_rows))
    selected_rows = available_rows.iloc[start_row:end_row]
    
    print(f"Processing {len(selected_rows)} rows:")
    for i, (_, row) in enumerate(selected_rows.iterrows()):
        print(f"  Row {start_row + i}: QID {row['QuestionId']} - {row['QuestionTitle'][:50]}...")
    
    results = []
    
    # Process each row
    for i, (_, row) in enumerate(selected_rows.iterrows()):
        row_num = start_row + i
        print(f"\n{'='*60}")
        print(f"Processing Row {row_num + 1}: QID {row['QuestionId']}")
        print(f"Title: {row['QuestionTitle']}")
        print(f"{'='*60}")
        
        # Clean the question data
        cleaned_question = clean_question_content(
            row['QuestionTitle'],
            row['QuestionBody'],
            row['QuestionTags']
        )
        
        question_data = {
            'question_id': row['QuestionId'],
            'title': cleaned_question['title'],
            'body': cleaned_question['body'],
            'tags': cleaned_question['tags']
        }
        
        # Generate answer with prompt 1
        print("Generating answer with Prompt 1...")
        result1 = client.generate_quantum_answer(question_data, examples, prompt1)
        
        # Combine results
        combined_result = {
            'row_number': row_num + 1,
            'question_id': row['QuestionId'],
            'question_title': row['QuestionTitle'],
            'question_body': row['QuestionBody'],
            'question_tags': row['QuestionTags'],
            'question_date': row['QuestionDate'],
            'accepted_answer_id': row['AcceptedAnswerId'],
            'answer_id': row['AnswerId'],
            'answer_body': row['Answer Body'],
            'answer_date': row['AnswerDate'],
            'answer_generated_by_q1': result1['answer'] if result1['success'] else '',
            'raw_response_prompt1': result1.get('raw_response', ''),
            'prompt1_success': result1['success'],
            'prompt1_error': result1.get('error', ''),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        results.append(combined_result)
        
        # Print status
        print(f"Prompt 1 success: {result1['success']}")
        if result1.get('error'):
            print(f"Prompt 1 error: {result1['error']}")
        
        if result1['success']:
            print(f"Answer preview: {result1['answer'][:100]}...")
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str) -> str:
    """
    Save results to CSV file with proper escaping for raw response columns
    """
    if not results:
        raise ValueError("No results to save")
    
    # Clean and escape raw response data before creating DataFrame
    cleaned_results = []
    for result in results:
        cleaned_result = result.copy()
        
        # Clean raw response columns - replace newlines and escape quotes
        for col in ['raw_response_prompt1']:
            if col in cleaned_result and cleaned_result[col]:
                # Replace newlines with spaces and escape quotes
                raw_text = str(cleaned_result[col])
                # Replace newlines with spaces
                raw_text = raw_text.replace('\n', ' ').replace('\r', ' ')
                # Escape double quotes by doubling them
                raw_text = raw_text.replace('"', '""')
                # Remove any remaining problematic characters
                raw_text = raw_text.replace('\t', ' ').replace('\0', '')
                # Clean up multiple spaces
                raw_text = ' '.join(raw_text.split())
                cleaned_result[col] = raw_text
        
        # Also clean other text columns that might have issues
        text_columns = ['question_title', 'question_body', 'answer_body', 'answer_generated_by_q1']
        for col in text_columns:
            if col in cleaned_result and cleaned_result[col]:
                text = str(cleaned_result[col])
                # Replace newlines with spaces
                text = text.replace('\n', ' ').replace('\r', ' ')
                # Escape double quotes
                text = text.replace('"', '""')
                # Remove tabs and null characters
                text = text.replace('\t', ' ').replace('\0', '')
                # Clean up multiple spaces
                text = ' '.join(text.split())
                cleaned_result[col] = text
        
        cleaned_results.append(cleaned_result)
    
    df = pd.DataFrame(cleaned_results)
    
    # Save with proper CSV quoting
    df.to_csv(output_file, index=False, quoting=1)  # quoting=1 means quote all fields
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED TO: {output_file}")
    print(f"{'='*60}")
    print(f"Total rows processed: {len(results)}")
    
    # Calculate success rates
    prompt1_success = sum(1 for r in results if r['prompt1_success'])
    
    print(f"Prompt 1 success rate: {prompt1_success}/{len(results)} ({prompt1_success/len(results)*100:.1f}%)")
    
    return output_file

def run_pipeline_with_tags_preserved(filtered_file: str, examples_file: str, 
                                    num_rows: int = 10, start_row: int = 0,
                                    prompt1_file: str = "prompt1.txt") -> List[Dict[str, Any]]:
    """
    Run pipeline with tags preserved (no tag cleaning)
    """
    print("="*60)
    print("RUNNING PIPELINE WITH TAGS PRESERVED")
    print("="*60)
    
    # Step 1: Load filtered dataset
    print("Step 1: Loading filtered dataset...")
    print(f"Using filtered dataset: {filtered_file}")
    
    # Step 2: Load examples
    print(f"\nStep 2: Loading few-shot examples...")
    examples = load_examples_from_json(examples_file)
    if not examples:
        raise ValueError("Could not load examples")
    
    example_qids = [ex['question_id'] for ex in examples]
    print(f"Loaded {len(examples)} examples with QIDs: {example_qids}")
    
    # Step 2: Load filtered dataset
    print(f"\nStep 2: Loading filtered dataset...")
    df = pd.read_csv(filtered_file)
    print(f"Dataset shape: {df.shape}")
    
    # Step 3: Load prompt template
    print(f"\nStep 3: Loading prompt template...")
    prompt1 = load_prompt_template(prompt1_file)
    
    if not prompt1:
        raise ValueError("Could not load prompt template")
    
    # Step 4: Initialize Ollama client
    print(f"\nStep 4: Initializing Ollama client...")
    client = OllamaDeepSeek()
    
    # Test connection
    test_result = client.test_connection()
    if not test_result['ollama_running'] or not test_result['model_available']:
        raise RuntimeError(f"Ollama not available: {test_result}")
    
    # Step 5: Select rows to process (excluding example rows)
    print(f"\nStep 5: Selecting rows {start_row} to {start_row + num_rows - 1} (excluding example rows)...")
    
    # Filter out example rows
    available_rows = df[~df['QuestionId'].isin(example_qids)].copy()
    print(f"Available rows after excluding examples: {len(available_rows)}")
    
    # Select the specified range
    end_row = min(start_row + num_rows, len(available_rows))
    selected_rows = available_rows.iloc[start_row:end_row]
    
    print(f"Processing {len(selected_rows)} rows:")
    for i, (_, row) in enumerate(selected_rows.iterrows()):
        print(f"  Row {start_row + i}: QID {row['QuestionId']} - {row['QuestionTitle'][:50]}...")
        print(f"    Tags: {row['QuestionTags']}")
    
    results = []
    
    # Step 6: Process each row
    for i, (_, row) in enumerate(selected_rows.iterrows()):
        row_num = start_row + i
        print(f"\n{'='*60}")
        print(f"Processing Row {row_num + 1}: QID {row['QuestionId']}")
        print(f"Title: {row['QuestionTitle']}")
        print(f"Tags: {row['QuestionTags']}")
        print(f"{'='*60}")
        
        # Clean the question data (but preserve tags)
        cleaned_question = clean_question_content(
            row['QuestionTitle'],
            row['QuestionBody'],
            row['QuestionTags']  # Keep tags as is
        )
        
        question_data = {
            'question_id': row['QuestionId'],
            'title': cleaned_question['title'],
            'body': cleaned_question['body'],
            'tags': cleaned_question['tags']  # This will preserve the original tags
        }
        
        # Generate answer with prompt 1
        print("Generating answer with Prompt 1...")
        result1 = client.generate_quantum_answer(question_data, examples, prompt1)
        
        # Combine results
        combined_result = {
            'row_number': row_num + 1,
            'question_id': row['QuestionId'],
            'question_title': row['QuestionTitle'],
            'question_body': row['QuestionBody'],
            'question_tags': row['QuestionTags'],  # Preserve original tags
            'question_date': row['QuestionDate'],
            'accepted_answer_id': row['AcceptedAnswerId'],
            'answer_id': row['AnswerId'],
            'answer_body': row['Answer Body'],
            'answer_date': row['AnswerDate'],
            'answer_generated_by_q1': result1['answer'] if result1['success'] else '',
            'raw_response_prompt1': result1.get('raw_response', ''),
            'prompt1_success': result1['success'],
            'prompt1_error': result1.get('error', ''),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        results.append(combined_result)
        
        # Print status
        print(f"Prompt 1 success: {result1['success']}")
        if result1.get('error'):
            print(f"Prompt 1 error: {result1['error']}")
        
        if result1['success']:
            print(f"Answer preview: {result1['answer'][:100]}...")
    
    return results

def main():
    """
    Main function
    """
    # Configuration
    filtered_file = "filtered_quantum_dataset_with_valid_answers.csv"
    examples_file = "few_shot_examples.json"
    num_rows = 3
    start_row = 0
    
    try:
        # Run pipeline on first 3 rows
        results = run_pipeline_on_first_n_rows(
            filtered_file=filtered_file,
            examples_file=examples_file,
            num_rows=num_rows,
            start_row=start_row
        )
        
        # Save results
        output_file = f"generated_answers_rows_{start_row}_{start_row + num_rows - 1}.csv"
        save_results(results, output_file)
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        raise

def main_with_tags():
    """
    Main function for running with tags preserved
    """
    # Configuration
    filtered_file = "filtered_quantum_dataset_with_valid_answers.csv"
    examples_file = "few_shot_examples.json"
    num_rows = 263
    start_row = 0
    
    try:
        # Run pipeline with tags preserved
        results = run_pipeline_with_tags_preserved(
            filtered_file=filtered_file,
            examples_file=examples_file,
            num_rows=num_rows,
            start_row=start_row
        )
        
        # Save results
        output_file = f"generated_answers.csv"
        save_results(results, output_file)
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {output_file}")
        
        # Show summary
        print(f"\nSUMMARY:")
        print(f"Total questions processed: {len(results)}")
        prompt1_success = sum(1 for r in results if r['prompt1_success'])
        print(f"Prompt 1 success: {prompt1_success}/{len(results)} ({prompt1_success/len(results)*100:.1f}%)")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    # Run with tags preserved
    main_with_tags()
