#!/usr/bin/env python3
"""
Ollama DeepSeek integration module for quantum Q&A generation
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List
from json_extraction import parse_llm_response

class OllamaDeepSeek:
    """
    Interface for interacting with Ollama DeepSeek model
    """
    
    def __init__(self, model_name: str = "deepseek-r1:32b", 
                 base_url: str = "http://localhost:11434", 
                 timeout: int = 180):
        """
        Initialize Ollama DeepSeek client
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.api_url = f"{base_url}/api/generate"
        
    def check_model_availability(self) -> bool:
        """
        Check if the model is available in Ollama
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            return self.model_name in model_names
            
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Generate response from DeepSeek model
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing response data
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result.get('response', ''),
                    'model': result.get('model', ''),
                    'created_at': result.get('created_at', ''),
                    'done': result.get('done', False),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'response': '',
                    'model': '',
                    'created_at': '',
                    'done': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'response': '',
                'model': '',
                'created_at': '',
                'done': False,
                'error': f"Request timeout after {self.timeout} seconds"
            }
        except Exception as e:
            return {
                'success': False,
                'response': '',
                'model': '',
                'created_at': '',
                'done': False,
                'error': f"Request failed: {str(e)}"
            }
    
    def generate_quantum_answer(self, question_data: Dict[str, str], 
                               few_shot_examples: List[Dict[str, Any]], 
                               prompt_template: str) -> Dict[str, Any]:
        """
        Generate answer for quantum computing question using few-shot prompting
        
        Args:
            question_data: Dictionary containing question information
            few_shot_examples: List of few-shot examples
            prompt_template: Prompt template to use
            
        Returns:
            Dictionary containing generated answer and metadata
        """
        # Create few-shot prompt
        from few_shot_selection import create_few_shot_prompt
        
        full_prompt = create_few_shot_prompt(
            few_shot_examples, 
            prompt_template, 
            question_data
        )
        
        # Generate response
        raw_response = self.generate_response(full_prompt)
        
        if not raw_response['success']:
            return {
                'success': False,
                'answer': '',
                'metadata': {},
                'raw_response': raw_response['response'],
                'error': raw_response['error']
            }
        
        # Parse the response
        parsed_response = parse_llm_response(raw_response['response'])
        
        return {
            'success': parsed_response['success'],
            'answer': parsed_response['answer'],
            'metadata': parsed_response['metadata'],
            'raw_response': raw_response['response'],
            'parsed_json': parsed_response['raw_json'],
            'error': parsed_response['error']
        }
    
    def batch_generate_answers(self, questions: List[Dict[str, Any]], 
                              few_shot_examples: List[Dict[str, Any]], 
                              prompt_template: str, 
                              delay_between_requests: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple questions
        
        Args:
            questions: List of question dictionaries
            few_shot_examples: Few-shot examples to use
            prompt_template: Prompt template
            delay_between_requests: Delay between requests in seconds
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}: {question.get('title', 'Unknown')[:50]}...")
            
            result = self.generate_quantum_answer(
                question, 
                few_shot_examples, 
                prompt_template
            )
            
            result['question_id'] = question.get('question_id', f'q_{i}')
            result['question_title'] = question.get('title', '')
            
            results.append(result)
            
            # Add delay between requests to avoid overwhelming the model
            if i < len(questions) - 1:
                time.sleep(delay_between_requests)
        
        return results
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Ollama and model availability
        """
        test_result = {
            'ollama_running': False,
            'model_available': False,
            'model_name': self.model_name,
            'error': None
        }
        
        try:
            # Test basic connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            test_result['ollama_running'] = (response.status_code == 200)
            
            if test_result['ollama_running']:
                # Check model availability
                test_result['model_available'] = self.check_model_availability()
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result

def load_prompt_template(file_path: str) -> str:
    """
    Load prompt template from file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file {file_path} not found")
        return ""
    except Exception as e:
        print(f"Error loading prompt file {file_path}: {e}")
        return ""

if __name__ == "__main__":
    # Test the Ollama integration
    print("Testing Ollama DeepSeek integration...")
    
    # Initialize client
    client = OllamaDeepSeek()
    
    # Test connection
    test_result = client.test_connection()
    print("Connection Test Results:")
    for key, value in test_result.items():
        print(f"{key}: {value}")
    
    if test_result['ollama_running'] and test_result['model_available']:
        print("\nTesting response generation...")
        
        # Test with a simple prompt
        test_prompt = "What is quantum computing? Please respond with a JSON object containing an 'answer' field."
        response = client.generate_response(test_prompt)
        
        if response['success']:
            print("Response generated successfully!")
            print(f"Response length: {len(response['response'])} characters")
            print(f"Response preview: {response['response'][:200]}...")
        else:
            print(f"Response generation failed: {response['error']}")
    else:
        print("\nCannot test response generation - Ollama not running or model not available")
        print("Please ensure Ollama is running and the deepseek-coder:32b model is installed")
