#!/usr/bin/env python3
"""
Robust JSON extraction utilities for parsing LLM responses
"""

import json
import re
from typing import Optional, Dict, Any, List

def robust_json_extraction(content: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from content using multiple methods
    """
    if not content or not isinstance(content, str):
        return None
        
    # Method 1: Clean and try direct parsing
    try:
        # Remove think blocks
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove markdown
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        # Try direct parse
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except:
        pass
    
    # Method 2: Find JSON patterns
    json_patterns = [
        r'\{[^{}]*"answer"[^{}]*"key_concepts"[^{}]*"difficulty_level"[^{}]*"related_topics"[^{}]*\}',
        r'\{[^{}]*"answer"[^{}]*"technical_depth"[^{}]*"core_principles"[^{}]*"practical_applications"[^{}]*"current_challenges"[^{}]*"future_implications"[^{}]*\}',
        r'\{[^{}]*"answer"[^{}]*\}',
        r'\{.*?"answer".*?\}',
        r'\{.*?\}(?=\s*$)',
        r'\{.*?\}',
    ]
    
    for pattern in json_patterns:
        try:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in reversed(matches):  # Try last match first (usually the final JSON)
                try:
                    result = json.loads(match)
                    if isinstance(result, dict) and 'answer' in result:
                        return result
                except:
                    continue
        except:
            continue
    
    # Method 3: Extract JSON with bracket counting
    try:
        start_idx = content.find('{')
        if start_idx != -1:
            bracket_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_str = content[start_idx:i+1]
                            try:
                                result = json.loads(json_str)
                                if isinstance(result, dict):
                                    return result
                            except:
                                pass
                            break
    except:
        pass
    
    # Method 4: Try to fix common JSON issues
    try:
        # Get content after last opening brace
        last_brace = content.rfind('{')
        if last_brace != -1:
            json_candidate = content[last_brace:]
            
            # Fix common issues
            json_candidate = re.sub(r',(\s*[}\]])', r'\1', json_candidate)  # Remove trailing commas
            json_candidate = re.sub(r'([}\]]),\s*$', r'\1', json_candidate)  # Remove final trailing comma
            
            try:
                result = json.loads(json_candidate)
                if isinstance(result, dict):
                    return result
            except:
                pass
    except:
        pass
    
    return None

def extract_answer_from_json(json_data: Dict[str, Any]) -> str:
    """
    Extract the answer text from parsed JSON data
    """
    if not isinstance(json_data, dict):
        return ""
    
    # Try different possible keys for the answer
    answer_keys = ['answer', 'response', 'text', 'content', 'result']
    
    for key in answer_keys:
        if key in json_data and isinstance(json_data[key], str):
            return json_data[key].strip()
    
    # If no answer key found, return the first string value
    for value in json_data.values():
        if isinstance(value, str) and len(value.strip()) > 10:
            return value.strip()
    
    return ""

def validate_json_response(json_data: Dict[str, Any], required_fields: List[str] = None) -> bool:
    """
    Validate that the JSON response contains required fields
    """
    if not isinstance(json_data, dict):
        return False
    
    if required_fields is None:
        required_fields = ['answer']
    
    for field in required_fields:
        if field not in json_data:
            return False
        
        if not isinstance(json_data[field], str) or len(json_data[field].strip()) < 5:
            return False
    
    return True

def clean_json_response(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and standardize JSON response data
    """
    if not isinstance(json_data, dict):
        return {}
    
    cleaned = {}
    
    # Clean string values
    for key, value in json_data.items():
        if isinstance(value, str):
            # Remove extra whitespace and newlines
            cleaned_value = re.sub(r'\s+', ' ', value).strip()
            cleaned[key] = cleaned_value
        elif isinstance(value, list):
            # Clean list items
            cleaned_list = []
            for item in value:
                if isinstance(item, str):
                    cleaned_list.append(re.sub(r'\s+', ' ', item).strip())
                else:
                    cleaned_list.append(item)
            cleaned[key] = cleaned_list
        else:
            cleaned[key] = value
    
    return cleaned

def extract_metadata_from_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata fields from JSON response
    """
    metadata = {}
    
    if not isinstance(json_data, dict):
        return metadata
    
    # Common metadata fields
    metadata_fields = {
        'key_concepts': 'key_concepts',
        'difficulty_level': 'difficulty_level',
        'related_topics': 'related_topics',
        'technical_depth': 'technical_depth',
        'core_principles': 'core_principles',
        'practical_applications': 'practical_applications',
        'current_challenges': 'current_challenges',
        'future_implications': 'future_implications'
    }
    
    for json_key, metadata_key in metadata_fields.items():
        if json_key in json_data:
            metadata[metadata_key] = json_data[json_key]
    
    return metadata

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response and extract structured data
    """
    result = {
        'success': False,
        'answer': '',
        'metadata': {},
        'raw_json': None,
        'error': None
    }
    
    try:
        # Extract JSON from response
        json_data = robust_json_extraction(response_text)
        
        if json_data is None:
            result['error'] = 'No valid JSON found in response'
            return result
        
        # Clean the JSON data
        cleaned_json = clean_json_response(json_data)
        result['raw_json'] = cleaned_json
        
        # Extract answer
        answer = extract_answer_from_json(cleaned_json)
        if not answer:
            result['error'] = 'No answer found in JSON response'
            return result
        
        result['answer'] = answer
        
        # Extract metadata
        metadata = extract_metadata_from_json(cleaned_json)
        result['metadata'] = metadata
        
        # Validate response
        if validate_json_response(cleaned_json):
            result['success'] = True
        else:
            result['error'] = 'JSON response missing required fields'
        
    except Exception as e:
        result['error'] = f'Error parsing response: {str(e)}'
    
    return result

if __name__ == "__main__":
    # Test the JSON extraction functions
    test_responses = [
        # Valid JSON response
        '{"answer": "This is a test answer", "key_concepts": ["concept1", "concept2"], "difficulty_level": "intermediate"}',
        
        # JSON with markdown
        '```json\n{"answer": "This is a test answer", "key_concepts": ["concept1", "concept2"]}\n```',
        
        # JSON with think tags
        '<think>Let me think about this...</think>{"answer": "This is a test answer", "key_concepts": ["concept1", "concept2"]}',
        
        # Malformed JSON
        '{"answer": "This is a test answer", "key_concepts": ["concept1", "concept2"],}',
        
        # No JSON
        'This is just plain text without any JSON structure.'
    ]
    
    print("Testing JSON extraction functions:")
    print("="*50)
    
    for i, response in enumerate(test_responses):
        print(f"\nTest {i+1}:")
        print(f"Input: {response[:100]}...")
        
        result = parse_llm_response(response)
        print(f"Success: {result['success']}")
        print(f"Answer: {result['answer'][:100]}...")
        if result['error']:
            print(f"Error: {result['error']}")
        print(f"Metadata: {result['metadata']}")

