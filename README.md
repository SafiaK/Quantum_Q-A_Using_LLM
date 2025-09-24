# Quantum Q&A Few-Shot Prompting Pipeline

A comprehensive pipeline for generating quantum computing answers using few-shot prompting with Ollama DeepSeek 32B model. This project processes a quantum computing Q&A dataset, filters out references, and generates new answers using two different prompt templates.

## 🚀 Features

- **Data Filtering**: Removes rows with href references and URLs
- **Text Cleaning**: Cleans HTML tags and normalizes text content
- **Few-Shot Learning**: Uses 2 examples from the dataset for context
- **Dual Prompt Templates**: Two different prompting strategies for comparison
- **Robust JSON Parsing**: Handles various response formats from the LLM
- **Semantic Analysis**: Advanced similarity analysis using sentence transformers
- **Answer Quality Assessment**: Comprehensive evaluation of generated answers
- **Raw Response Capture**: Stores complete LLM responses for analysis
- **Modular Design**: Separate modules for each pipeline component
- **Ollama Integration**: Works with local Ollama DeepSeek model

## 📁 Project Structure

```
├── data_filtering.py              # Dataset filtering and preprocessing
├── text_cleaning.py               # Text cleaning utilities
├── few_shot_selection.py          # Example selection for few-shot learning
├── select_examples.py             # Script to select and save examples
├── ollama_integration.py          # Ollama DeepSeek model integration
├── json_extraction.py             # Robust JSON response parsing
├── run_pipeline.py                # Main pipeline execution script
├── run_first_10.py                # Script to run on first 10 rows
├── main_pipeline.py               # Original comprehensive pipeline
├── quantum_qa_analysis.py         # Comprehensive data analysis utility
├── prompt1.txt                    # First prompt template
├── prompt2.txt                    # Second prompt template
├── few_shot_examples.json         # Selected few-shot examples
├── filtered_quantum_dataset.csv   # Filtered dataset (no href references)
├── generated_answers_first_10_rows.csv  # Generated answers output
├── generated_answers_with_tags_first_200_rows.csv  # Full dataset results
├── quantum_qa_analysis_report.txt # Analysis report with similarity scores
└── requirements.txt               # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- Ollama installed and running
- DeepSeek model available in Ollama

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SafiaK/Quantum_Q-A_Using_LLM.git
   cd Quantum_Q-A_Using_LLM
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama:**
   ```bash
   # Install Ollama (macOS)
   brew install ollama
   
   # Or download from https://ollama.ai/
   
   # Start Ollama service
   ollama serve
   
   # Pull DeepSeek model
   ollama pull deepseek-r1:32b
   ```

4. **Verify installation:**
   ```bash
   python3 ollama_integration.py
   ```



## 🔧 Usage

### 1. Data Preprocessing

Filter the dataset to remove rows with href references:

```bash
python3 data_filtering.py
```

This creates `filtered_quantum_dataset.csv` with clean data.

### 2. Select Few-Shot Examples

Select 2 examples from the filtered dataset:

```bash
python3 select_examples.py
```

This creates `few_shot_examples.json` with selected examples.

### 3. Run Pipeline

Generate answers for the first 10 rows:

```bash
python3 run_first_10.py
```

This creates `generated_answers_first_10_rows.csv` with generated answers.

### 4. Run Analysis

Analyze the generated answers with semantic similarity:

```bash
python3 quantum_qa_analysis.py
```

This creates `quantum_qa_analysis_report.txt` with detailed analysis including:
- Success rate analysis
- Semantic similarity scores using sentence transformers
- Answer quality assessment
- Prompt comparison statistics

## 📊 Analysis Features

### Semantic Similarity Analysis
- Uses sentence-transformers (all-MiniLM-L6-v2) for accurate semantic comparison
- Compares generated answers with original Stack Overflow answers
- Provides similarity scores between 0-1 (higher = more similar)
- Handles different writing styles and phrasings

### Success Rate Analysis
- Raw response generation rates
- JSON parsing success rates
- Answer extraction from raw responses
- Combined success metrics

### Quality Assessment
- Average similarity scores per prompt
- Similarity range and distribution
- Prompt comparison (which performs better)
- Extraction success rates

## 📋 Pipeline Components

### 1. Data Filtering (`data_filtering.py`)

- Removes rows containing href references, URLs, and external links
- Cleans HTML tags while preserving content
- Handles multiple text encodings (UTF-8, Latin-1, CP1252, ISO-8859-1)
- Generates filtering statistics

### 2. Text Cleaning (`text_cleaning.py`)

- Removes HTML tags and entities
- Normalizes whitespace and formatting
- Extracts key quantum computing concepts
- Validates cleaned content quality

### 3. Few-Shot Selection (`select_examples.py`)

- Selects diverse examples based on question tags
- Ensures examples meet quality criteria (length, content)
- Saves examples in JSON format for reuse
- Excludes selected examples from target questions

### 4. Ollama Integration (`ollama_integration.py`)

- Interfaces with local Ollama service
- Handles model communication and error handling
- Supports configurable timeouts and parameters
- Tests model availability and connectivity

### 5. JSON Extraction (`json_extraction.py`)

- Robust parsing of LLM responses
- Handles various JSON formats and malformed responses
- Extracts structured data from free-form text
- Multiple fallback parsing methods
