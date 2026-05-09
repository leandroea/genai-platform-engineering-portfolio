# Job Search Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.0.0+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.0+-green.svg)

A multi-agent CLI system that autonomously manages a user's job search campaign. Coordinates independent agents to find jobs, tailor application materials, track applications, and prepare for interviews — all through a conversational interface.

## Problem Solved

Job seekers waste time on repetitive tasks (customizing resumes, writing cover letters, finding jobs) and miss opportunities due to disorganization. This system automates the tedium while maintaining personalization, using AI agents that work independently under a supervisor's coordination.

## Key Features

- **Multi-Agent Architecture**: Supervisor coordinates 6 specialized subordinate agents
- **Resume Tailoring**: Customize resumes for specific job applications (PDF/DOCX input, DOCX output)
- **Job Aggregation**: Search jobs via Jooble API with relevance scoring
- **Resume Scoring**: Evaluate resume-job match quality with actionable recommendations
- **Cover Letter Generation**: Personalized letters based on job descriptions
- **Application Form Assistance**: Auto-fill form fields with resume-derived content
- **Interview Preparation**: Generate practice questions with STAR-method answers
- **Chat Mode**: Natural language interaction for all operations
- **Persistent State**: Save and resume job search campaigns

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Supervisor Agent](#supervisor-agent)
  - [Resume Tailor Agent](#resume-tailor-agent)
  - [Cover Letter Agent](#cover-letter-agent)
  - [Job Aggregator Agent](#job-aggregator-agent)
  - [Resume Score Checker Agent](#resume-score-checker-agent)
  - [Interview Coach Agent](#interview-coach-agent)
  - [Application Form Agent](#application-form-agent)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
  - [Menu Options](#menu-options)

---

## Project Structure

```
job-search-assistant/
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── main.py                  # CLI entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py        # Supervisor agent (coordinator)
│   │   ├── resume_tailor.py     # Resume customization agent
│   │   ├── cover_letter.py      # Cover letter writer agent
│   │   ├── job_aggregator.py    # Job search agent (Jooble API)
│   │   ├── resume_scorer.py     # Resume scoring agent
│   │   ├── application_form.py   # Form field completion agent
│   │   └── interview_coach.py   # Interview preparation agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── resume_tools.py      # PDF/DOCX parsing and writing
│   │   ├── job_tools.py         # Jooble API integration
│   │   ├── io_tools.py          # CLI I/O helpers
│   │   └── job_description_tools.py  # Local job description management
│   ├── state/
│   │   ├── __init__.py
│   │   └── job_search_state.py  # Shared state definition
│   ├── workflow/
│   │   ├── __init__.py
│   │   └── job_search_graph.py  # LangGraph workflow
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py      # User intent detection
│   │   ├── conversation_context.py   # Conversation memory
│   │   └── conversational_agent.py    # Natural language interface
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Environment variable loader
├── data/
│   ├── resume/                  # User uploads original resume here
│   ├── output/                  # Generated resumes, cover letters
│   └── job_descriptions/        # Local job description storage
└── tests/
    ├── __init__.py
    ├── test_resume_tools.py
    ├── test_job_tools.py
    └── test_workflow.py
```

---

## Architecture

The system uses **LangGraph** for workflow orchestration with a **Supervisor pattern** where a central Supervisor agent coordinates independent subordinate agents. Unlike sequential pipelines, agents work autonomously and report results back to the Supervisor for decision-making.

### Agent Coordination Model

The Supervisor agent acts as the central coordinator, dispatching tasks to independent subordinate agents and synthesizing their results:

```
                            ┌─────────────────────┐
                            │   SUPERVISOR AGENT  │
                            │   (Coordinator)     │
                            └──────────┬──────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┬──────────────┐
        │              │               │               │              │              │
        ▼              ▼               ▼               ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      ┌───────────┐
   │ Resume  │    │  Cover  │    │   Job   │    │  Resume │    │Interview│      │Application│
   │  Tailor │    │  Letter │    │Aggregator│   │  Scorer │    │  Coach  │      │   Form    │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘      └───────────┘
                                            
```

**Key Pattern**: Unlike sequential pipelines, agents work **autonomously** and report back to the Supervisor for decision-making.

---

### Supervisor Agent

**Role**: Coordinator and Decision Maker

**Behavior**:
- Receives user requests and decides which agents to activate
- Queries each subordinate agent for their current status and outputs
- Synthesizes information from multiple agents to make decisions
- Prioritizes tasks when multiple urgent items exist
- Does NOT route sequentially — coordinates independently working agents

**Example decisions**:
- "Resume Score is 45/100 — too low. Ask Resume Tailor to improve before applying"
- "Found 3 high-matching jobs — prioritize these for immediate application"
- "Cover letter generated but score is low — regenerate with more personalization"

**States managed**: `current_phase`, `pending_tasks`, `completed_tasks`, `blockers`

---

### Resume Tailor Agent

**Role**: Resume Customization Specialist

**Behavior**:
- Receives original resume text and target job description
- Analyzes job requirements and identifies gaps in resume
- Rewrites resume content to highlight relevant skills/experience
- Maintains truthfulness — only emphasizes existing experience

**Operations**:

| Operation | Description |
|-----------|-------------|
| Full Tailor | Rewrites entire resume for a job posting |
| Section Edit | Modifies specific section (e.g., "update my experience to highlight Python") |
| Keyword Inject | Adds missing keywords without changing content |
| Format Adjust | Changes formatting, order, or layout |

**Output**: DOCX file saved to `data/output/resume_{company}_{job_id}.docx`

---

### Cover Letter Agent

**Role**: Personalized Letter Writer

**Behavior**:
- Receives job details and user's background summary
- Researches company culture and values from job description
- Writes professional cover letter with personalization
- Follows standard business letter structure

**What it does**:
- Creates engaging opening paragraph with specific job reference
- Highlights 2-3 most relevant qualifications
- Connects user's experience to company's needs
- Closes with call-to-action and gratitude

---

### Job Aggregator Agent

**Role**: Job Search and Discovery

**Behavior**:
- Receives target roles, locations, and preferences
- Calls Jooble API to search for relevant job postings
- Deduplicates results and filters by relevance
- Returns structured job listings with metadata

**Output**:
```python
{
    "jobs": [
        {
            "title": "Senior Python Developer", 
            "company": "TechCorp", 
            "location": "Remote", 
            "url": "https://jooble.org/...",
            "salary_min": 120000,
            "match_score": 0.87
        },
        ...
    ],
    "search_query_used": "Senior Python Developer Remote"
}
```

---

### Resume Score Checker Agent

**Role**: Quality Assurance for Applications

**Behavior**:
- Receives resume text and job description
- Evaluates match quality across multiple criteria
- Provides numerical score with breakdown
- Suggests specific improvements

**Scoring Criteria**:

| Category | Points | What it Measures |
|----------|--------|------------------|
| Keyword Match | 0-25 | Job-specific terms present in resume |
| Experience Relevance | 0-25 | Past roles match job requirements |
| Format & Length | 0-25 | Clean formatting, 1-2 pages |
| Achievements | 0-25 | Quantified results, not just duties |

**Example output**:
```
Overall Score: 68/100
- Keyword Match: 18/25 (Python, AWS, Django found)
- Experience Relevance: 15/25 (2/3 requirements matched)
- Format & Length: 20/25 (good formatting, slightly long)
- Achievements: 15/25 (need more quantified results)

Recommendations:
1. Add "Led team of 5 engineers" to highlight leadership
2. Include "30% performance improvement" metrics
3. Remove "Microsoft Office" — irrelevant for this role
```

---

### Interview Coach Agent

**Role**: Interview Preparation Specialist

**Behavior**:
- Receives job description and company research
- Generates relevant interview questions
- Prepares STAR-method answers for behavioral questions
- Runs mock interview sessions on demand

**What it does**:
- Generates 10-15 likely technical questions based on job
- Creates 5-7 behavioral questions (STAR format)
- Provides sample answers and talking points
- Conducts mock Q&A sessions
- Tracks which questions have been practiced

---

### Application Form Agent

**Role**: Form Field Completion Assistant

**Behavior**:
- Receives requests to fill application form fields
- Extracts relevant information from user's resume
- Generates appropriate responses for specific platforms
- Handles both simple fields (name, email) and complex ones (work history, skills)

**Supported Field Types**:

| Field Type | Example |
|------------|---------|
| Personal | Name, email, phone, address, LinkedIn URL |
| Professional | Job title, company, dates, salary expectations |
| Education | Degree, institution, graduation year, GPA |
| Skills | Technical skills, languages, certifications |
| Free Text | "Why do you want to work here?", "Describe your experience" |

---

## Usage

### Prerequisites

- **Python 3.11+**
- **MiniMax API Key** (or OpenAI-compatible API)
- **Jooble API Key** (free tier available at [jooble.org](https://jooble.org))

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd job-search-assistant
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** and add your API keys:
   ```env
   MINIMAX_API_KEY=your-minimax-api-key
   MINIMAX_ENDPOINT=https://api.minimax.io/v1  # OpenAI-compatible endpoint
   MODEL_NAME=minimax-m2.7
   JOOBLE_API_KEY=your-jooble-api-key
   ```

### Running the Application

```bash
python -m src.main
```

Or from project root:
```bash
python src/main.py
```

### Menu Options

| Option | Description |
|--------|-------------|
| `1` | Upload Resume (PDF or DOCX) |
| `2` | Set Job Preferences (roles, locations, companies) |
| `3` | Search for Jobs (via Jooble API) |
| `4` | Score Resume Against Job |
| `5` | Generate Tailored Resume |
| `6` | Fill Application Form |
| `7` | Generate Cover Letter |
| `8` | Interview Preparation |
| `9` | View Saved Jobs |
| `A` | Add Job Description (manual entry) |
| `B` | Manage Local Job Descriptions |
| `C` | Upload Job Description File |
| `T` | Chat Mode (Natural Language) |
| `0` | Exit |

---

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | LangChain | >= 1.0.0 |
| Graph Engine | LangGraph | >= 1.0.0 |
| LLM | ChatOpenAI-compatible (MiniMax API) | minimax-m2.7 |
| Python | Python | >= 3.11 |
| Job Search API | Jooble API | Free tier |
| Resume Parsing (PDF) | pdfplumber | >= 0.11.0 |
| Resume Parsing (DOCX) | python-docx | >= 1.1.0 |
| Resume Writing (DOCX) | python-docx | >= 1.1.0 |
| Environment | python-dotenv | >= 1.0.0 |
| HTTP Client | requests | >= 2.31.0 |

---