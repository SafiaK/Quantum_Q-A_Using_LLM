#!/usr/bin/env python3
"""
Main pipeline for quantum Q&A answer generation using Ollama DeepSeek
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# Import our custom modules
from data_filtering import filter_dataset, analyze_dataset
from text_cleaning import clean_question_content, clean_answer_content
from few_shot_selection import select_diverse_examples, analyze_example_quality
from ollama_integration import OllamaDeepSeek, load_prompt_template
from json_extraction import parse_llm_response

class QuantumQAPipeline:
    """
    Main pipeline for quantum Q&A answer generation
    """
    
    def __init__(self, model_name: str = "deepseek-r1:32b"):
        """
        Initialize the pipeline
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.ollama_client = OllamaDeepSeek(model_name)
        self.results = []
        
    def setup_pipeline(self, input_file: str, filtered_file: str = "filtered_quantum_dataset.csv") -> Dict[str, Any]:
        """
        Set up the pipeline by filtering the dataset
        
        Args:
            input_file: Path to input CSV file
            filtered_file: Path to save filtered dataset
            
        Returns:
            Dictionary containing setup statistics
        """
        print("Setting up pipeline...")
        print("="*50)
        
        # Filter the dataset
        stats = filter_dataset(input_file, filtered_file)
        
        # Analyze the filtered dataset
        analysis = analyze_dataset(filtered_file)
        
        setup_stats = {
            'filtering_stats': stats,
            'analysis_stats': analysis,
            'filtered_file': filtered_file
        }
        
        print(f"Pipeline setup complete. Filtered dataset saved to {filtered_file}")
        return setup_stats
    
    def select_questions_for_generation(self, filtered_file: str, num_questions: int = 10) -> List[Dict[str, Any]]:
        """
        Select questions for answer generation
        
        Args:
            filtered_file: Path to filtered dataset
            num_questions: Number of questions to select
            
        Returns:
            List of selected questions
        """
        print(f"\nSelecting {num_questions} questions for generation...")
        
        df = pd.read_csv(filtered_file)
        
        # Filter for good questions (not too short, not too long)
        good_questions = df[
            (df['QuestionTitle'].str.len() >= 20) &
            (df['QuestionBody'].str.len() >= 50) &
            (df['Answer Body'].str.len() >= 100)
        ].copy()
        
        if len(good_questions) < num_questions:
            print(f"Warning: Only {len(good_questions)} questions meet criteria")
            num_questions = len(good_questions)
        
        # Select questions with multiple answers if available
        question_counts = good_questions.groupby('QuestionId').size()
        multi_answer_questions = question_counts[question_counts > 1].index.tolist()
        
        selected_questions = []
        selected_qids = set()
        
        # First, try to select questions with multiple answers
        for qid in multi_answer_questions:
            if len(selected_questions) >= num_questions:
                break
            
            question_rows = good_questions[good_questions['QuestionId'] == qid]
            if len(question_rows) > 0:
                # Take the first answer for this question
                row = question_rows.iloc[0]
                
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
                    'tags': cleaned_question['tags'],
                    'question_date': row['QuestionDate'],
                    'accepted_answer_id': row['AcceptedAnswerId'],
                    'answer_id': row['AnswerId'],
                    'answer_body': clean_answer_content(row['Answer Body']),
                    'answer_date': row['AnswerDate']
                }
                
                selected_questions.append(question_data)
                selected_qids.add(qid)
        
        # Fill remaining slots with random questions
        remaining_needed = num_questions - len(selected_questions)
        if remaining_needed > 0:
            remaining_questions = good_questions[~good_questions['QuestionId'].isin(selected_qids)]
            if len(remaining_questions) > 0:
                sample_size = min(remaining_needed, len(remaining_questions))
                sample_questions = remaining_questions.sample(sample_size)
                
                for _, row in sample_questions.iterrows():
                    cleaned_question = clean_question_content(
                        row['QuestionTitle'],
                        row['QuestionBody'],
                        row['QuestionTags']
                    )
                    
                    question_data = {
                        'question_id': row['QuestionId'],
                        'title': cleaned_question['title'],
                        'body': cleaned_question['body'],
                        'tags': cleaned_question['tags'],
                        'question_date': row['QuestionDate'],
                        'accepted_answer_id': row['AcceptedAnswerId'],
                        'answer_id': row['AnswerId'],
                        'answer_body': clean_answer_content(row['Answer Body']),
                        'answer_date': row['AnswerDate']
                    }
                    
                    selected_questions.append(question_data)
        
        print(f"Selected {len(selected_questions)} questions for generation")
        return selected_questions
    
    def generate_answers(self, questions: List[Dict[str, Any]], 
                        few_shot_examples: List[Dict[str, Any]], 
                        prompt1_file: str = "prompt1.txt", 
                        prompt2_file: str = "prompt2.txt") -> List[Dict[str, Any]]:
        """
        Generate answers for selected questions using both prompts
        
        Args:
            questions: List of questions to process
            few_shot_examples: Few-shot examples
            prompt1_file: Path to first prompt template
            prompt2_file: Path to second prompt template
            
        Returns:
            List of results with generated answers
        """
        print(f"\nGenerating answers for {len(questions)} questions...")
        print("="*50)
        
        # Load prompt templates
        prompt1 = load_prompt_template(prompt1_file)
        prompt2 = load_prompt_template(prompt2_file)
        
        if not prompt1 or not prompt2:
            raise ValueError("Could not load prompt templates")
        
        # Test Ollama connection
        test_result = self.ollama_client.test_connection()
        if not test_result['ollama_running'] or not test_result['model_available']:
            raise RuntimeError(f"Ollama not available: {test_result}")
        
        results = []
        
        for i, question in enumerate(questions):
            print(f"\nProcessing question {i+1}/{len(questions)}: {question['title'][:50]}...")
            
            # Generate answer with prompt 1
            print("  Generating answer with prompt 1...")
            result1 = self.ollama_client.generate_quantum_answer(
                question, few_shot_examples, prompt1
            )
            
            # Generate answer with prompt 2
            print("  Generating answer with prompt 2...")
            result2 = self.ollama_client.generate_quantum_answer(
                question, few_shot_examples, prompt2
            )
            
            # Combine results
            combined_result = {
                'question_id': question['question_id'],
                'question_title': question['title'],
                'question_body': question['body'],
                'question_tags': question['tags'],
                'question_date': question['question_date'],
                'accepted_answer_id': question['accepted_answer_id'],
                'answer_id': question['answer_id'],
                'answer_body': question['answer_body'],
                'answer_date': question['answer_date'],
                'answer_generated_by_q1': result1['answer'] if result1['success'] else '',
                'answer_generated_by_q2': result2['answer'] if result2['success'] else '',
                'prompt1_success': result1['success'],
                'prompt2_success': result2['success'],
                'prompt1_error': result1.get('error', ''),
                'prompt2_error': result2.get('error', ''),
                'prompt1_metadata': result1.get('metadata', {}),
                'prompt2_metadata': result2.get('metadata', {}),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            results.append(combined_result)
            
            # Print status
            print(f"  Prompt 1 success: {result1['success']}")
            print(f"  Prompt 2 success: {result2['success']}")
            if result1.get('error'):
                print(f"  Prompt 1 error: {result1['error']}")
            if result2.get('error'):
                print(f"  Prompt 2 error: {result2['error']}")
        
        self.results = results
        return results
    
    def save_results(self, output_file: str = "generated_answers.csv") -> str:
        """
        Save generated results to CSV file
        
        Args:
            output_file: Path to save results
            
        Returns:
            Path to saved file
        """
        if not self.results:
            raise ValueError("No results to save. Run generate_answers first.")
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Total questions processed: {len(self.results)}")
        
        # Calculate success rates
        prompt1_success = sum(1 for r in self.results if r['prompt1_success'])
        prompt2_success = sum(1 for r in self.results if r['prompt2_success'])
        
        print(f"Prompt 1 success rate: {prompt1_success}/{len(self.results)} ({prompt1_success/len(self.results)*100:.1f}%)")
        print(f"Prompt 2 success rate: {prompt2_success}/{len(self.results)} ({prompt2_success/len(self.results)*100:.1f}%)")
        
        return output_file
    
    def run_full_pipeline(self, input_file: str, num_questions: int = 10, 
                         num_examples: int = 2) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            input_file: Path to input CSV file
            num_questions: Number of questions to process
            num_examples: Number of few-shot examples to use
            
        Returns:
            Dictionary containing pipeline results
        """
        print("Starting Quantum Q&A Pipeline")
        print("="*50)
        
        # Step 1: Setup pipeline
        setup_stats = self.setup_pipeline(input_file)
        
        # Step 2: Select few-shot examples
        print(f"\nSelecting {num_examples} few-shot examples...")
        df = pd.read_csv(setup_stats['filtered_file'])
        few_shot_examples = select_diverse_examples(df, num_examples)
        
        example_analysis = analyze_example_quality(few_shot_examples)
        print(f"Selected {len(few_shot_examples)} examples")
        print(f"Example quality analysis: {example_analysis}")
        
        # Step 3: Select questions for generation
        questions = self.select_questions_for_generation(
            setup_stats['filtered_file'], 
            num_questions
        )
        
        # Step 4: Generate answers
        results = self.generate_answers(questions, few_shot_examples)
        
        # Step 5: Save results
        output_file = self.save_results()
        
        pipeline_results = {
            'setup_stats': setup_stats,
            'example_analysis': example_analysis,
            'questions_processed': len(questions),
            'results': results,
            'output_file': output_file,
            'success_rates': {
                'prompt1': sum(1 for r in results if r['prompt1_success']) / len(results),
                'prompt2': sum(1 for r in results if r['prompt2_success']) / len(results)
            }
        }
        
        return pipeline_results

def main():
    """
    Main function to run the pipeline
    """
    # Configuration
    input_file = "Quantum_Dataset 26-06-2024.csv"
    num_questions = 10
    num_examples = 2
    
    # Initialize pipeline
    pipeline = QuantumQAPipeline()
    
    try:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            input_file=input_file,
            num_questions=num_questions,
            num_examples=num_examples
        )
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Questions processed: {results['questions_processed']}")
        print(f"Output file: {results['output_file']}")
        print(f"Prompt 1 success rate: {results['success_rates']['prompt1']*100:.1f}%")
        print(f"Prompt 2 success rate: {results['success_rates']['prompt2']*100:.1f}%")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
