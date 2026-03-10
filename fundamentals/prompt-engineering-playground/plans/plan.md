# Prompt Engineering Playground - Project Plan

## Overview
A Streamlit web application for testing, comparing, and evaluating LLM prompts with experiment history tracking.

## Tech Stack
- **Python** - Programming language
- **Streamlit** - Web UI framework
- **SQLite** - Database for experiment history
- **LangChain** - LLM integration framework
- **NVIDIA API** - LLM provider (Llama 3.3 70B)

## Project Structure
```
prompt-engineering-playground/
├── .env                    # API configuration (already exists)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── venv/                   # Virtual environment
├── app/
│   ├── __init__.py
│   ├── main.py             # Streamlit app entry point
│   ├── llm_service.py      # LLM API integration
│   ├── database.py         # SQLite operations
│   ├── models.py           # Data models
│   └── utils.py            # Utility functions
├── data/
│   └── experiments.db      # SQLite database
└── plans/
    └── plan.md             # This plan
```

## Features

### 1. Single Prompt Testing
- Text area for entering prompts
- Adjustable LLM parameters:
  - Temperature (0.0 - 2.0)
  - Max tokens (1 - 4096)
  - Top-p (0.0 - 1.0)
  - Frequency penalty (-2.0 - 2.0)
  - Presence penalty (-2.0 - 2.0)
- Display LLM response
- Save to experiment history with optional notes

### 2. Prompt Comparison
- Compare 2-3 prompts side by side
- Same parameters for all prompts
- Display responses in columns
- Save comparison as single experiment

### 3. Manual Evaluation
- Rate outputs (1-5 stars or thumbs up/down)
- Add text feedback notes
- Categorize by quality dimensions:
  - Accuracy
  - Helpfulness
  - Clarity

### 4. Experiment History
- List all past experiments
- Filter by date, prompt text, rating
- View experiment details
- Delete experiments
- Re-run experiments with modified parameters

## Database Schema

### experiments table
- id: INTEGER PRIMARY KEY
- created_at: DATETIME
- prompt: TEXT
- parameters: TEXT (JSON)
- response: TEXT
- rating: INTEGER (1-5)
- feedback: TEXT
- experiment_type: TEXT (single/comparison)
- name: TEXT (optional)

### comparison_results table
- id: INTEGER PRIMARY KEY
- experiment_id: INTEGER (FK)
- prompt_index: INTEGER
- response: TEXT
- rating: INTEGER

## UI Layout

### Navigation
- Sidebar with sections:
  1. Test Prompt
  2. Compare Prompts
  3. History

### Main Area
- Dynamic based on selected section
- Consistent header with app title

## Implementation Steps

1. **Setup** - Create venv, install dependencies
2. **Database** - Create SQLite schema and operations
3. **LLM Service** - Implement LangChain + NVIDIA integration
4. **UI - Test Prompt** - Single prompt testing page
5. **UI - Compare** - Side-by-side comparison page
6. **UI - History** - Experiment history with filtering
7. **Testing** - Verify all features work
