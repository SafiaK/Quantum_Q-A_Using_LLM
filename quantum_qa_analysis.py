#!/usr/bin/env python3
"""
Quantum Q&A Data Analysis Utility

This module provides comprehensive analysis of the generated quantum Q&A dataset,
including success rate analysis, answer extraction from raw responses, and
similarity comparison between original and generated answers.
"""

import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np
from collections import Counter

class QuantumQAAnalyzer:
    """
    Comprehensive analyzer for quantum Q&A dataset
    """
    
    def __init__(self, csv_file: str = "generated_answers_with_tags_first_200_rows.csv"):
        """
        Initialize the analyzer with the dataset
        
        Args:
            csv_file: Path to the generated answers CSV file
        """
        self.csv_file = csv_file
        self.df = None
        self.sentence_model = None
        self.load_data()
        self.setup_sentence_transformer()
    
    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Dataset loaded successfully: {len(self.df)} rows")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def setup_sentence_transformer(self):
        """Setup sentence transformer model for semantic similarity"""
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Load a pre-trained sentence transformer model
            # Using 'all-MiniLM-L6-v2' which is fast and good for general text similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.cosine_similarity = cosine_similarity
            
            print("✅ Sentence transformer model loaded successfully")
            print(f"Model: {self.sentence_model.get_sentence_embedding_dimension()}D embeddings")
            
        except ImportError:
            print("⚠️ Warning: sentence-transformers not available")
            print("Installing sentence-transformers...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity
                
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.cosine_similarity = cosine_similarity
                print("✅ Sentence transformer model installed and loaded successfully")
                
            except Exception as e:
                print(f"❌ Could not install sentence-transformers: {e}")
                print("Falling back to basic similarity calculation")
                self.sentence_model = None
                
        except Exception as e:
            print(f"⚠️ Warning: Could not setup sentence transformer: {e}")
            print("Falling back to basic similarity calculation")
            self.sentence_model = None
    
    def analyze_success_rates(self) -> Dict:
        """
        Analyze success rates for both prompts based on JSON response availability
        
        Returns:
            Dictionary containing detailed success rate analysis
        """
        print("\n" + "="*60)
        print("📊 SUCCESS RATE ANALYSIS")
        print("="*60)
        
        # Count total rows
        total_rows = len(self.df)
        
        # Count successful responses (non-empty raw responses)
        prompt1_success = self.df['raw_response_prompt1'].notna() & (self.df['raw_response_prompt1'] != '')
        prompt2_success = self.df['raw_response_prompt2'].notna() & (self.df['raw_response_prompt2'] != '')
        
        # Count JSON parsing success (based on success flags)
        prompt1_json_success = self.df['prompt1_success'].sum()
        prompt2_json_success = self.df['prompt2_success'].sum()
        
        # Calculate rates
        prompt1_raw_rate = prompt1_success.sum() / total_rows * 100
        prompt2_raw_rate = prompt2_success.sum() / total_rows * 100
        prompt1_json_rate = prompt1_json_success / total_rows * 100
        prompt2_json_rate = prompt2_json_success / total_rows * 100
        
        # Combined success (at least one prompt successful)
        combined_success = (prompt1_success | prompt2_success).sum()
        combined_rate = combined_success / total_rows * 100
        
        analysis = {
            'total_rows': total_rows,
            'prompt1_raw_responses': prompt1_success.sum(),
            'prompt2_raw_responses': prompt2_success.sum(),
            'prompt1_json_success': prompt1_json_success,
            'prompt2_json_success': prompt2_json_success,
            'prompt1_raw_rate': prompt1_raw_rate,
            'prompt2_raw_rate': prompt2_raw_rate,
            'prompt1_json_rate': prompt1_json_rate,
            'prompt2_json_rate': prompt2_json_rate,
            'combined_success': combined_success,
            'combined_rate': combined_rate
        }
        
        # Print results
        print(f"Total rows processed: {total_rows}")
        print(f"\nRaw Response Analysis:")
        print(f"  Prompt 1 raw responses: {prompt1_success.sum()}/{total_rows} ({prompt1_raw_rate:.1f}%)")
        print(f"  Prompt 2 raw responses: {prompt2_success.sum()}/{total_rows} ({prompt2_raw_rate:.1f}%)")
        print(f"\nJSON Parsing Success:")
        print(f"  Prompt 1 JSON success: {prompt1_json_success}/{total_rows} ({prompt1_json_rate:.1f}%)")
        print(f"  Prompt 2 JSON success: {prompt2_json_success}/{total_rows} ({prompt2_json_rate:.1f}%)")
        print(f"\nCombined Success:")
        print(f"  At least one successful: {combined_success}/{total_rows} ({combined_rate:.1f}%)")
        
        return analysis
    
    def extract_answer_from_raw_response(self, raw_response: str) -> Optional[str]:
        """
        Extract answer from raw response by removing <think> tags and parsing JSON
        
        Args:
            raw_response: Raw response string from LLM
            
        Returns:
            Extracted answer string or None if extraction fails
        """
        if not raw_response or pd.isna(raw_response):
            return None
        
        try:
            # Remove <think> tags and content
            cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                
                # Try different possible keys for the answer
                answer_keys = ['answer', 'response', 'content', 'text', 'explanation']
                for key in answer_keys:
                    if key in parsed_json and parsed_json[key]:
                        return str(parsed_json[key]).strip()
            
            # If no JSON found, return the cleaned response
            return cleaned_response.strip()
            
        except Exception as e:
            print(f"Warning: Could not extract answer from raw response: {e}")
            return None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for semantic similarity calculation
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and clean
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate very long texts (sentence transformers work better with reasonable lengths)
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text strings using sentence transformers
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2 or pd.isna(text1) or pd.isna(text2):
            return 0.0
        
        # Preprocess texts
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)
        
        if not processed_text1 or not processed_text2:
            return 0.0
        
        # Use sentence transformer if available
        if self.sentence_model:
            try:
                # Encode the texts to get embeddings
                embeddings = self.sentence_model.encode([processed_text1, processed_text2])
                
                # Calculate cosine similarity between embeddings
                similarity = self.cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # Sentence transformer is working correctly
                
                return float(similarity)
                
            except Exception as e:
                print(f"Warning: Sentence transformer similarity failed, falling back to basic similarity: {e}")
                # Fallback to basic similarity
                return SequenceMatcher(None, processed_text1, processed_text2).ratio()
        else:
            print("Warning: No sentence transformer model available, using basic similarity")
            # Fallback to basic similarity
            return SequenceMatcher(None, processed_text1, processed_text2).ratio()
    
    def analyze_answer_quality(self) -> Dict:
        """
        Analyze answer quality by comparing generated answers with original answers
        
        Returns:
            Dictionary containing quality analysis results
        """
        print("\n" + "="*60)
        print("🔍 ANSWER QUALITY ANALYSIS")
        print("="*60)
        
        results = {
            'prompt1_analysis': {},
            'prompt2_analysis': {},
            'comparison_stats': {}
        }
        
        # Analyze Prompt 1
        print("\nAnalyzing Prompt 1 answers...")
        prompt1_results = self._analyze_prompt_answers('prompt1', 'raw_response_prompt1', 'answer_generated_by_q1')
        results['prompt1_analysis'] = prompt1_results
        
        # Analyze Prompt 2
        print("\nAnalyzing Prompt 2 answers...")
        prompt2_results = self._analyze_prompt_answers('prompt2', 'raw_response_prompt2', 'answer_generated_by_q2')
        results['prompt2_analysis'] = prompt2_results
        
        # Comparison statistics
        print("\nGenerating comparison statistics...")
        comparison_stats = self._generate_comparison_stats(prompt1_results, prompt2_results)
        results['comparison_stats'] = comparison_stats
        
        return results
    
    def _analyze_prompt_answers(self, prompt_name: str, raw_col: str, answer_col: str) -> Dict:
        """
        Analyze answers for a specific prompt
        
        Args:
            prompt_name: Name of the prompt (for display)
            raw_col: Column name for raw response
            answer_col: Column name for generated answer
            
        Returns:
            Dictionary containing analysis results
        """
        similarities = []
        extraction_success = 0
        total_processed = 0
        similarity_calculations = 0
        
        print(f"  Processing {len(self.df)} total rows...")
        
        for idx, row in self.df.iterrows():
            if pd.isna(row[raw_col]) or row[raw_col] == '':
                continue
            
            total_processed += 1
            
            # Get original answer
            original_answer = row['answer_body'] if not pd.isna(row['answer_body']) else ''
            
            # Try to get answer from generated column first
            generated_answer = row[answer_col] if not pd.isna(row[answer_col]) else ''
            
            # If no generated answer, try to extract from raw response
            if not generated_answer:
                generated_answer = self.extract_answer_from_raw_response(row[raw_col])
                if generated_answer:
                    extraction_success += 1
            
            # Calculate similarity for ALL rows with both answers
            if generated_answer and original_answer:
                similarity = self.calculate_similarity(original_answer, generated_answer)
                similarities.append(similarity)
                similarity_calculations += 1
                
                # Print progress every 50 calculations
                if similarity_calculations % 50 == 0:
                    print(f"    Processed {similarity_calculations} similarity calculations...")
        
        print(f"  Total processed: {total_processed}")
        print(f"  Similarity calculations: {similarity_calculations}")
        
        # Calculate statistics
        if similarities:
            avg_similarity = np.mean(similarities)
            median_similarity = np.median(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
            std_similarity = np.std(similarities)
        else:
            avg_similarity = median_similarity = min_similarity = max_similarity = std_similarity = 0.0
        
        extraction_rate = extraction_success / total_processed * 100 if total_processed > 0 else 0
        
        results = {
            'total_processed': total_processed,
            'extraction_success': extraction_success,
            'extraction_rate': extraction_rate,
            'similarities': similarities,
            'avg_similarity': avg_similarity,
            'median_similarity': median_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'std_similarity': std_similarity
        }
        
        print(f"  Total processed: {total_processed}")
        print(f"  Extraction success: {extraction_success} ({extraction_rate:.1f}%)")
        print(f"  Average similarity: {avg_similarity:.3f}")
        print(f"  Median similarity: {median_similarity:.3f}")
        print(f"  Similarity range: {min_similarity:.3f} - {max_similarity:.3f}")
        
        return results
    
    def _generate_comparison_stats(self, prompt1_results: Dict, prompt2_results: Dict) -> Dict:
        """
        Generate comparison statistics between the two prompts
        
        Args:
            prompt1_results: Analysis results for prompt 1
            prompt2_results: Analysis results for prompt 2
            
        Returns:
            Dictionary containing comparison statistics
        """
        stats = {
            'prompt1_better_similarity': 0,
            'prompt2_better_similarity': 0,
            'tie_similarity': 0,
            'avg_similarity_diff': 0,
            'prompt1_extraction_rate': prompt1_results['extraction_rate'],
            'prompt2_extraction_rate': prompt2_results['extraction_rate']
        }
        
        # Compare similarities where both prompts have data
        sim1 = prompt1_results['similarities']
        sim2 = prompt2_results['similarities']
        
        if sim1 and sim2:
            min_len = min(len(sim1), len(sim2))
            if min_len > 0:
                for i in range(min_len):
                    if sim1[i] > sim2[i]:
                        stats['prompt1_better_similarity'] += 1
                    elif sim2[i] > sim1[i]:
                        stats['prompt2_better_similarity'] += 1
                    else:
                        stats['tie_similarity'] += 1
                
                stats['avg_similarity_diff'] = np.mean(sim1[:min_len]) - np.mean(sim2[:min_len])
        
        print(f"\nComparison Statistics:")
        print(f"  Prompt 1 better similarity: {stats['prompt1_better_similarity']}")
        print(f"  Prompt 2 better similarity: {stats['prompt2_better_similarity']}")
        print(f"  Tie similarity: {stats['tie_similarity']}")
        print(f"  Average similarity difference: {stats['avg_similarity_diff']:.3f}")
        print(f"  Prompt 1 extraction rate: {stats['prompt1_extraction_rate']:.1f}%")
        print(f"  Prompt 2 extraction rate: {stats['prompt2_extraction_rate']:.1f}%")
        
        return stats
    
    def generate_detailed_report(self) -> str:
        """
        Generate a comprehensive analysis report
        
        Returns:
            Formatted report string
        """
        print("\n" + "="*60)
        print("📋 GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        # Run all analyses
        success_analysis = self.analyze_success_rates()
        quality_analysis = self.analyze_answer_quality()
        
        # Generate report
        report = f"""
# Quantum Q&A Dataset Analysis Report

## Dataset Overview
- **Total Rows**: {success_analysis['total_rows']}
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Success Rate Analysis

### Raw Response Availability
- **Prompt 1**: {success_analysis['prompt1_raw_responses']}/{success_analysis['total_rows']} ({success_analysis['prompt1_raw_rate']:.1f}%)
- **Prompt 2**: {success_analysis['prompt2_raw_responses']}/{success_analysis['total_rows']} ({success_analysis['prompt2_raw_rate']:.1f}%)

### JSON Parsing Success
- **Prompt 1**: {success_analysis['prompt1_json_success']}/{success_analysis['total_rows']} ({success_analysis['prompt1_json_rate']:.1f}%)
- **Prompt 2**: {success_analysis['prompt2_json_success']}/{success_analysis['total_rows']} ({success_analysis['prompt2_json_rate']:.1f}%)

### Combined Success
- **At least one successful**: {success_analysis['combined_success']}/{success_analysis['total_rows']} ({success_analysis['combined_rate']:.1f}%)

## Answer Quality Analysis

### Prompt 1 Quality Metrics
- **Total Processed**: {quality_analysis['prompt1_analysis']['total_processed']}
- **Answer Extraction Success**: {quality_analysis['prompt1_analysis']['extraction_success']} ({quality_analysis['prompt1_analysis']['extraction_rate']:.1f}%)
- **Average Similarity to Original**: {quality_analysis['prompt1_analysis']['avg_similarity']:.3f}
- **Median Similarity**: {quality_analysis['prompt1_analysis']['median_similarity']:.3f}
- **Similarity Range**: {quality_analysis['prompt1_analysis']['min_similarity']:.3f} - {quality_analysis['prompt1_analysis']['max_similarity']:.3f}

### Prompt 2 Quality Metrics
- **Total Processed**: {quality_analysis['prompt2_analysis']['total_processed']}
- **Answer Extraction Success**: {quality_analysis['prompt2_analysis']['extraction_success']} ({quality_analysis['prompt2_analysis']['extraction_rate']:.1f}%)
- **Average Similarity to Original**: {quality_analysis['prompt2_analysis']['avg_similarity']:.3f}
- **Median Similarity**: {quality_analysis['prompt2_analysis']['median_similarity']:.3f}
- **Similarity Range**: {quality_analysis['prompt2_analysis']['min_similarity']:.3f} - {quality_analysis['prompt2_analysis']['max_similarity']:.3f}

### Prompt Comparison
- **Prompt 1 Better Similarity**: {quality_analysis['comparison_stats']['prompt1_better_similarity']} cases
- **Prompt 2 Better Similarity**: {quality_analysis['comparison_stats']['prompt2_better_similarity']} cases
- **Tie Similarity**: {quality_analysis['comparison_stats']['tie_similarity']} cases
- **Average Similarity Difference**: {quality_analysis['comparison_stats']['avg_similarity_diff']:.3f}


---
*Report generated by Quantum Q&A Analysis Utility with Sentence Transformers*
"""
        
        return report
    
    def save_report(self, filename: str = "quantum_qa_analysis_report.txt"):
        """
        Save the analysis report to a file
        
        Args:
            filename: Name of the file to save the report
        """
        report = self.generate_detailed_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Analysis report saved to: {filename}")

def main():
    """
    Main function to run the complete analysis
    """
    print("🚀 Starting Quantum Q&A Dataset Analysis")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = QuantumQAAnalyzer()
        
        # Run complete analysis
        analyzer.save_report()
        
        print("\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
