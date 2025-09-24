#!/usr/bin/env python3
"""
Text cleaning utilities for quantum Q&A data
"""

import re
import html
from typing import Optional, Dict, Any

def clean_html_content(text: str) -> str:
    """
    Clean HTML content while preserving meaningful text
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags but preserve content
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def clean_question_content(question_title: str, question_body: str, question_tags: str) -> Dict[str, str]:
    """
    Clean and format question content
    """
    cleaned_title = clean_html_content(question_title)
    cleaned_body = clean_html_content(question_body)
    cleaned_tags = clean_html_content(question_tags)
    
    # Format tags properly
    if cleaned_tags:
        # Remove angle brackets and split by >
        tags = re.sub(r'[<>]', '', cleaned_tags).split('>')
        tags = [tag.strip() for tag in tags if tag.strip()]
        cleaned_tags = ', '.join(tags)
    
    return {
        'title': cleaned_title,
        'body': cleaned_body,
        'tags': cleaned_tags
    }

def clean_answer_content(answer_body: str) -> str:
    """
    Clean and format answer content
    """
    return clean_html_content(answer_body)

def format_question_for_prompt(question_data: Dict[str, str]) -> str:
    """
    Format question data for use in prompts
    """
    title = question_data.get('title', '')
    body = question_data.get('body', '')
    tags = question_data.get('tags', '')
    
    formatted = f"Title: {title}\n"
    if body:
        formatted += f"Body: {body}\n"
    if tags:
        formatted += f"Tags: {tags}\n"
    
    return formatted.strip()

def format_answer_for_prompt(answer_body: str) -> str:
    """
    Format answer data for use in prompts
    """
    return clean_answer_content(answer_body)

def extract_key_concepts(text: str) -> list:
    """
    Extract key concepts from text (quantum computing related terms)
    """
    if not isinstance(text, str):
        return []
    
    # Quantum computing related terms
    quantum_terms = [
        'quantum', 'qubit', 'superposition', 'entanglement', 'decoherence',
        'quantum gate', 'quantum circuit', 'quantum algorithm', 'quantum error correction',
        'quantum annealing', 'quantum supremacy', 'quantum advantage', 'quantum cryptography',
        'quantum key distribution', 'quantum teleportation', 'quantum interference',
        'quantum tunneling', 'quantum coherence', 'quantum measurement',
        'quantum state', 'quantum mechanics', 'quantum physics'
    ]
    
    found_terms = []
    text_lower = text.lower()
    
    for term in quantum_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return list(set(found_terms))

def validate_cleaned_content(content: str, min_length: int = 10) -> bool:
    """
    Validate that cleaned content meets minimum requirements
    """
    if not isinstance(content, str):
        return False
    
    # Check minimum length
    if len(content.strip()) < min_length:
        return False
    
    # Check for meaningful content (not just whitespace or special characters)
    meaningful_chars = re.sub(r'[^\w\s]', '', content)
    if len(meaningful_chars.strip()) < min_length // 2:
        return False
    
    return True

def create_question_summary(question_data: Dict[str, str]) -> str:
    """
    Create a concise summary of the question for few-shot examples
    """
    title = question_data.get('title', '')
    body = question_data.get('body', '')
    tags = question_data.get('tags', '')
    
    # Truncate body if too long
    if len(body) > 200:
        body = body[:200] + "..."
    
    summary = f"Q: {title}"
    if body:
        summary += f"\nContext: {body}"
    if tags:
        summary += f"\nTags: {tags}"
    
    return summary

def create_answer_summary(answer_body: str, max_length: int = 300) -> str:
    """
    Create a concise summary of the answer for few-shot examples
    """
    cleaned_answer = clean_answer_content(answer_body)
    
    if len(cleaned_answer) > max_length:
        cleaned_answer = cleaned_answer[:max_length] + "..."
    
    return f"A: {cleaned_answer}"

if __name__ == "__main__":
    # Test the cleaning functions
    test_question = {
        'title': 'What is quantum computing?',
        'body': '<p>I want to understand the basics of quantum computing and how it differs from classical computing.</p>',
        'tags': '<quantum-computing><basics>'
    }
    
    test_answer = '<p>Quantum computing is a type of computation that uses quantum mechanical phenomena like superposition and entanglement to process information.</p>'
    
    print("Testing text cleaning functions:")
    print("="*50)
    
    cleaned_q = clean_question_content(
        test_question['title'],
        test_question['body'],
        test_question['tags']
    )
    
    print("Cleaned question:")
    print(format_question_for_prompt(cleaned_q))
    print()
    
    print("Cleaned answer:")
    print(format_answer_for_prompt(test_answer))
    print()
    
    print("Question summary:")
    print(create_question_summary(cleaned_q))
    print()
    
    print("Answer summary:")
    print(create_answer_summary(test_answer))
