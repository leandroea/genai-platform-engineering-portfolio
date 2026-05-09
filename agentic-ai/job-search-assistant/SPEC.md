# Job Search Assistant - Specification

> Persistent memory for AI agents working on the Job Search Assistant project.
> This document prevents context drift and ensures coherent code generation.

---

## 1. Project Overview & Context

### Purpose

The **Job Search Assistant** is a multi-agent CLI system that autonomously manages a user's job search campaign. It coordinates independent subordinate agents to find jobs, tailor application materials, track applications, and prepare for interviews — all without requiring a web interface or browser.

**Problem Solved**: Job seekers waste time on repetitive tasks (customizing resumes, writing cover letters, finding jobs) and miss opportunities due to disorganization. This system automates the tedium while maintaining personalization.

### Goal

A working CLI application where:
- A **Supervisor Agent** coordinates strategy and makes decisions
- **Resume Tailor Agent** customizes resumes per job application
- **Cover Letter Agent** writes personalized letters
- **Job Aggregator Agent** finds relevant job postings
- **Resume Score Checker Agent** scores resumes against job descriptions
- **Application Form Agent** assists with manual form field completion
- **Interview Coach Agent** prepares interview questions and practice

**Success Criteria**:
- [ ] Supervisor coordinates all 6 subordinate agents
- [ ] Subordinates work independently (not in a pipeline)
- [ ] Supervisor can query each subordinate's state and outputs
- [ ] User interacts via CLI only (no browser, no web server)
- [ ] Original resume can be PDF or DOCX format
- [ ] Tailored resumes are saved in DOCX format
- [ ] All API integrations use real connections (no mocks in tests)

### Agent Descriptions

#### Supervisor Agent
**Role**: Coordinator and Decision Maker

**Behavior**:
- Receives user requests and decides which agents to activate
- Queries each subordinate agent for their current status and outputs
- Synthesizes information from multiple agents to make decisions
- Prioritizes tasks when multiple urgent items exist
- Does NOT route sequentially (not a pipeline) — coordinates independently working agents

**What it does**:
- Interprets user intent and dispatches appropriate agents
- Evaluates job match quality based on agent outputs
- Decides application strategy (which jobs to prioritize)
- Handles conflicts when agent recommendations disagree
- Maintains overall job search campaign state

**Example decisions**:
- "Resume Score is 45/100 — too low. Ask Resume Tailor to improve before applying"
- "Found 3 high-matching jobs — prioritize these for immediate application"
- "Cover letter generated but score is low — regenerate with more personalization"

---

#### Resume Tailor Agent
**Role**: Resume Customization Specialist

**Behavior**:
- Receives original resume text and target job description
- Analyzes job requirements and identifies gaps in resume
- Rewrites resume content to highlight relevant skills/experience
- Can modify specific sections upon request (not just full resume)
- Maintains truthfulness — only emphasizes existing experience

**What it does**:
- Extracts key skills from job description
- Maps user experience to job requirements
- Optimizes keyword density for ATS (Applicant Tracking Systems)
- Restructures resume sections for maximum impact
- Modifies specific sections when user requests targeted changes
- Generates tailored resume as DOCX file

**Operations**:
| Operation | Description |
|-----------|-------------|
| Full Tailor | Rewrites entire resume for a job posting |
| Section Edit | Modifies specific section (e.g., "update my experience to highlight Python") |
| Keyword Inject | Adds missing keywords without changing content |
| Format Adjust | Changes formatting, order, or layout |

**Example operations**:
```
User: "Tailor my resume for the Senior Python Developer job"
→ Full resume rewrite targeting the job description

User: "Update my skills section to emphasize leadership"
→ Only the skills section is modified, rest unchanged

User: "Add AWS and Docker to my experience section"
→ Keywords added to experience, truthful to actual experience
```

---

#### Cover Letter Agent
**Role**: Personalized Letter Writer

**Behavior**:
- Receives job details and user's background summary
- Researches company culture and values (from job description)
- Writes professional cover letter with personalization
- Follows standard business letter structure

**What it does**:
- Creates engaging opening paragraph with specific job reference
- Highlights 2-3 most relevant qualifications
- Connects user's experience to company's needs
- Closes with call-to-action and gratitude
- Saves cover letter as DOCX (optional)

**Example output**:
```
Dear Hiring Manager,
When I read about the Product Manager role at Acme Corp, I immediately saw how my background in B2B SaaS aligns with your team's focus on enterprise customers...
```

---

#### Job Aggregator Agent
**Role**: Job Search and Discovery

**Behavior**:
- Receives target roles, locations, and preferences
- Calls Jooble API to search for relevant job postings
- Deduplicates results and filters by relevance
- Returns structured job listings with metadata

**What it does**:
- Searches Jooble API with keywords and location filters
- Extracts job title, company, location, salary range, URL
- Scores jobs by relevance to user preferences
- Returns up to 20 jobs per search
- Handles API rate limits gracefully

**Example output**:
```python
{
    "jobs": [
        {"title": "Senior Python Developer", 
         "company": "TechCorp", 
         "location": "Remote", 
         "url": "https://jooble.org/...",
         "salary_min": 120000,
         "match_score": 0.87},
        ...
    ],
    "search_query_used": "Senior Python Developer Remote"
}
```

---

#### Resume Score Checker Agent
**Role**: Quality Assurance for Applications

**Behavior**:
- Receives resume text and job description
- Evaluates match quality across multiple criteria
- Provides numerical score with breakdown
- Suggests specific improvements

**What it does**:
- Scores keyword match (0-25 points)
- Scores experience relevance (0-25 points)  
- Scores format and length (0-25 points)
- Scores achievements and metrics (0-25 points)
- Generates improvement recommendations
- Returns overall score 0-100 with category breakdown

**Scoring criteria**:
| Category | Points | What it measures |
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

#### Interview Coach Agent
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

**Example output**:
```
Technical Questions:
1. "Explain the difference between SQL and NoSQL databases"
2. "How would you design a scalable API?"
3. "Describe your experience with cloud deployment"

Behavioral Questions (STAR):
1. "Tell me about a time you resolved a technical conflict"
   - Situation: Team disagreed on architecture choice
   - Task: Reach consensus without delaying project
   - Action: Organized proof-of-concept demo
   - Result: Team chose approach, project delivered 2 weeks early
```

---

#### Application Form Agent
**Role**: Form Field Completion Assistant

**Behavior**:
- Receives requests to fill application form fields
- Extracts relevant information from user's resume
- Generates appropriate responses for specific platforms
- Handles both simple fields (name, email) and complex ones (work history, skills)

**What it does**:
- Parses user resume to extract all relevant information
- Maps resume data to form field labels
- Handles platform-specific formatting requirements
- Generates suggestions for free-text fields (work history, achievements)
- Provides copy-paste ready answers

**Operations**:
| Operation | Description |
|-----------|-------------|
| Field Suggestion | Provide answer for a specific field (e.g., "Years of Experience") |
| Full Form Fill | Generate all field answers for a platform |
| Work History Entry | Format work experience for application |
| Skills Translation | Map resume skills to platform skill fields |

**Supported Field Types**:
| Field Type | Example |
|------------|---------|
| Personal | Name, email, phone, address, LinkedIn URL |
| Professional | Job title, company, dates, salary expectations |
| Education | Degree, institution, graduation year, GPA |
| Skills | Technical skills, languages, certifications |
| Free Text | "Why do you want to work here?", "Describe your experience" |

**Example output**:
```
Platform: LinkedIn Easy Apply
Fields:
- Current Title: "Software Engineer" (from resume)
- Years of Experience: "5" (calculated from work history)
- Skills: "Python, JavaScript, AWS, Docker, SQL" (extracted from resume)
- Work Authorization: "US Citizen / Green Card holder"
- LinkedIn URL: (provided by user)

Free-text suggestions:
- "Tell us about yourself": "Experienced software engineer with 5+ years..."
- "Why this company?": Generated based on job description analysis
```

---

## 2. Technical Stack and Rules

### Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Framework** | LangChain | >= 1.0.0 |
| **Graph Engine** | LangGraph | >= 1.0.0 |
| **LLM** | ChatOpenAI-compatible (MiniMax API) | minimax-m2.7 |
| **Python** | Python | >= 3.11 |
| **Job Search API** | Jooble API | Free tier |
| **Resume Parsing (PDF)** | pdfplumber | >= 0.11.0 |
| **Resume Parsing (DOCX)** | python-docx | >= 1.1.0 |
| **Resume Writing (DOCX)** | python-docx | >= 1.1.0 |
| **Environment** | python-dotenv | >= 1.0.0 |

### Dependencies (requirements.txt)

```
langchain
langgraph
langchain-openai
python-docx
pdfplumber
python-dotenv
```

### Rules & Constraints

| Rule | Description |
|------|-------------|
| **Architecture** | Must use LangGraph for workflow orchestration |
| **Agent Pattern** | Supervisor coordinates; subordinates work independently |
| **State Management** | All agent outputs stored in shared `JobSearchState` |
| **No Hardcoding** | All API keys, endpoints from environment variables |
| **CLI Only** | No web server, no browser interface |
| **Resume Format Input** | Support PDF (.pdf) and DOCX (.docx) |
| **Resume Format Output** | Tailored resumes saved as DOCX (.docx) |
| **No Mocks in Tests** | All API/LLM calls use real connections |

### Version Pinning

```python
# .env file structure
MINIMAX_API_KEY=your-minimax-api-key
MINIMAX_ENDPOINT=https://api.minimax.io/v1  # OpenAI-compatible
MODEL_NAME=minimax-m2.7
JOOBLE_API_KEY=your-jooble-api-key  # Free tier available
```

---

## 3. Data Model and Architecture

### Data Schema

#### JobSearchState (TypedDict)

```python
class JobSearchState(TypedDict):
    """Shared state passed between supervisor and all agents."""
    
    # User Profile
    user_profile: dict              # Name, email, phone, location, experience summary
    original_resume_path: str       # Path to uploaded resume (PDF or DOCX)
    original_resume_text: str       # Extracted text from resume
    
    # Job Search Parameters
    target_roles: list[str]        # e.g., ["Product Manager", "Senior Developer"]
    target_locations: list[str]     # e.g., ["Remote", "New York"]
    target_companies: list[str]     # Optional: specific companies to target
    
    # Agent Outputs (each agent writes here)
    resume_tailor_output: dict     # {tailored_resume_docx_path, job_id}
    cover_letter_output: dict       # {cover_letter_text, job_id}
    job_aggregator_output: dict     # {jobs: list[job], search_query_used}
    resume_score_output: dict       # {score, breakdown, recommendations}
    application_form_output: dict   # {field_answers, platform_type, job_id}
    interview_coach_output: dict    # {questions: list, answers: list, mock_session}
    
    # Supervisor Decision State
    current_phase: str              # "job_search", "application", "interview_prep", "idle"
    pending_tasks: list[str]       # Tasks the supervisor has assigned
    completed_tasks: list[str]      # Tasks completed by subordinates
    blockers: list[str]             # Issues that need attention
    supervisor_decision: str        # Current decision/reasoning from supervisor
```

#### Job (Data Model)

```python
class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    company: str
    location: str
    description: str
    url: str
    salary_min: Optional[int]
    salary_max: Optional[int]
    posted_date: Optional[str]
    source: str = "jooble"  # API source
    match_score: float = 0.0  # Calculated by supervisor
    applied: bool = False
    applied_date: Optional[datetime]
```

#### Application (Data Model)

```python
class Application(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    tailored_resume_path: str
    cover_letter_path: Optional[str]
    status: Literal["pending", "applied", "interview", "rejected", "offer"] = "pending"
    applied_date: Optional[datetime]
    notes: str = ""
    interview_date: Optional[datetime]
    feedback: str = ""
```

### File Structure

```
job-search-assistant/    # Project root (already inside this folder)
├── SPEC.md                    # This file
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys)
├── .gitignore                 # Exclude .env, output/
├── src/
│   ├── __init__.py
│   ├── main.py               # CLI entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py     # Supervisor agent
│   │   ├── resume_tailor.py  # Resume customization
│   │   ├── cover_letter.py   # Cover letter writer
│   │   ├── job_aggregator.py # Job search
│   │   ├── resume_scorer.py  # Resume scoring against job description
│   │   ├── application_form.py # Form field completion assistance
│   │   └── interview_coach.py # Interview prep
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── resume_tools.py   # PDF/DOCX parsing and writing
│   │   ├── job_tools.py      # Jooble API integration
│   │   └── io_tools.py       # CLI I/O helpers
│   ├── state/
│   │   ├── __init__.py
│   │   └── job_search_state.py
│   ├── workflow/
│   │   ├── __init__.py
│   │   └── job_search_graph.py  # LangGraph workflow
│   └── utils/
│       ├── __init__.py
│       └── config.py         # Environment variable loader
├── data/
│   ├── resume/               # User uploads original resume here
│   └── output/              # Generated resumes, cover letters
└── tests/
    ├── __init__.py
    ├── test_resume_tools.py
    ├── test_job_tools.py
    └── test_workflow.py
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Files | snake_case | `job_search_graph.py` |
| Classes | PascalCase | `JobSearchState` |
| Functions | snake_case | `extract_resume_text` |
| Constants | UPPER_SNAKE | `MAX_APPLICATIONS` |
| Environment | UPPER_SNAKE | `MINIMAX_API_KEY` |

### Resume File Handling

#### Input Resume (User Provides)
- **PDF**: `.pdf` file → extracted using `pdfplumber`
- **DOCX**: `.docx` file → extracted using `python-docx`

#### Output Resume (Generated)
- **DOCX**: `.docx` file → created using `python-docx`
- Format: Tailored for specific job application
- Naming: `resume_{company_slug}_{job_id}.docx`

#### Resume Text Extraction Flow
```
User uploads: resume.pdf or resume.docx
    ↓
pdfplumber/python-docx extracts text
    ↓
Text stored in JobSearchState.original_resume_text
    ↓
LLM reads text to understand background
    ↓
Resume Tailor creates tailored content
    ↓
python-docx writes new .docx file
    ↓
Saved to data/output/resume_{company}_{job_id}.docx
```

---

## 4. Implementation Plan

### Phase Breakdown

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Foundation | Project setup, state definition, CLI interface | 1-2 days |
| 2 | Resume Handling | PDF/DOCX parsing, text extraction, DOCX generation | 1-2 days |
| 3 | Job Aggregator | Jooble API integration, job search | 1-2 days |
| 4 | Resume Tailor | Customizing resume for specific jobs | 1-2 days |
| 5 | Resume Score Checker | Scoring resumes against job descriptions | 1-2 days |
| 6 | Application Form Agent | Form field completion assistance | 1-2 days |
| 7 | Cover Letter Agent | Writing personalized letters | 1-2 days |
| 8 | Supervisor + Interview Coach | Orchestration and interview prep | 1-2 days |
| 9 | Integration & Testing | Full workflow, real API tests | 2-3 days |

### Task List

#### Phase 1: Foundation
- [ ] Create project structure (folders, __init__.py files)
- [ ] Set up requirements.txt with pinned versions
- [ ] Create .env template with all required variables
- [ ] Define JobSearchState TypedDict
- [ ] Create CLI main.py with menu system
- [ ] Verify environment variables load correctly

#### Phase 2: Resume Handling
- [ ] Create resume parsing function (PDF via pdfplumber)
- [ ] Create resume parsing function (DOCX via python-docx)
- [ ] Create resume writing function (DOCX via python-docx)
- [ ] Create utility to detect file format from extension
- [ ] Test with sample PDF and DOCX files
- [ ] Verify text extraction accuracy

#### Phase 3: Job Aggregator
- [ ] Create Jooble API client
- [ ] Implement job search by keyword, location
- [ ] Implement pagination (max 20 results per search)
- [ ] Create job deduplication logic
- [ ] Test with real Jooble API call
- [ ] Handle API rate limits gracefully

#### Phase 4: Resume Tailor
- [ ] Create supervisor instruction template for tailoring
- [ ] Implement resume analysis (what to highlight)
- [ ] Create job-specific keyword optimization
- [ ] Implement skills matching logic
- [ ] Generate tailored resume DOCX
- [ ] Test with different job descriptions

#### Phase 5: Resume Score Checker
- [ ] Define scoring criteria (keywords, experience match, format)
- [ ] Create scoring prompt for LLM
- [ ] Implement resume-vs-job-description comparison
- [ ] Generate score breakdown (0-100)
- [ ] Provide specific improvement recommendations
- [ ] Test with sample job descriptions

#### Phase 6: Application Form Agent
- [ ] Create form field parsing system
- [ ] Implement resume-to-field mapping
- [ ] Add platform-specific formatting
- [ ] Create free-text field suggestions
- [ ] Test with sample job applications

#### Phase 7: Cover Letter Agent
- [ ] Create cover letter generation prompt
- [ ] Implement personalization based on job/company
- [ ] Generate professional letter structure
- [ ] Save cover letter as DOCX (optional)
- [ ] Test with sample job posting

#### Phase 8: Supervisor + Interview Coach
- [ ] Create Supervisor agent with coordination logic
- [ ] Implement supervisor decision-making
- [ ] Add Interview Coach with question generation
- [ ] Create mock interview functionality
- [ ] Implement supervisor-subordinate communication
- [ ] Test supervisor decisions with real state

#### Phase 7: Integration & Testing
- [ ] Create end-to-end workflow test
- [ ] Test all agents with real LLM calls
- [ ] Test Jooble API with real connection
- [ ] Test resume parsing with real files
- [ ] Handle all edge cases
- [ ] Document CLI usage instructions

### Edge Cases

| Scenario | Handling |
|----------|----------|
| **Resume file not found** | Prompt user to upload valid file |
| **Unsupported resume format** | Only accept .pdf or .docx, reject others |
| **Jooble API returns 0 jobs** | Suggest broadening search terms |
| **Jooble API rate limit** | Wait 60 seconds, retry with exponential backoff |
| **LLM API timeout** | Retry 3 times, then save state and exit |
| **Invalid JSON in application DB** | Backup corrupted file, start fresh |
| **Empty job description** | Skip job, log warning |
| **Resume too long** | Truncate to 2-page equivalent |

---

## 5. AI Guardrails

### "Never" Rules

| Never Do This | Reason |
|---------------|--------|
| **Never delete SPEC.md** | This is the project source of truth |
| **Never delete .env** | Contains secrets, recreate from template only |
| **Never commit API keys** | Violates security, use gitignore |
| **Never hardcode credentials** | All secrets from environment variables |
| **Never skip error handling** | Production code must handle failures gracefully |
| **Never use mocks in tests** | Must test with real API connections |
| **Never skip resume file validation** | Must verify file exists and is readable |
| **Never overwrite original resume** | Always save tailored as new file |

### Safety Constraints

| Constraint | Implementation |
|------------|----------------|
| **API key protection** | Use `os.getenv()` only, never string literals |
| **Rate limiting** | Add delays between API calls, respect Jooble limits |
| **Input sanitization** | Escape special characters in user inputs |
| **File validation** | Check file extension before processing |
| **Backup before write** | Before creating DOCX, verify path is valid |
| **Graceful degradation** | If one agent fails, others continue |

### Files That Must Never Be Deleted

```
SPEC.md
.env
requirements.txt
src/agents/supervisor.py
src/state/job_search_state.py
src/workflow/job_search_graph.py
src/tools/resume_tools.py
```

---

## 6. Verification and Testing

### Acceptance Criteria

| # | Criterion | Test Method |
|---|-----------|-------------|
| 1 | CLI launches and displays menu | Run `python src/main.py` |
| 2 | Resume PDF parsed correctly | Upload test.pdf, verify text extracted |
| 3 | Resume DOCX parsed correctly | Upload test.docx, verify text extracted |
| 4 | Tailored resume saved as DOCX | Generate resume, verify .docx created |
| 5 | Jooble API returns real jobs | Call API with "Software Engineer", verify results |
| 6 | Resume scored correctly | Score a resume against job description, verify breakdown |
| 7 | Cover letter generated | Generate letter, verify personalization |
| 8 | Application form filled | Provide field suggestions, verify accuracy |
| 9 | Supervisor queries all agents | Run workflow, verify agent outputs in state |
| 10 | Interview questions generated | Ask for interview prep, verify questions |
| 11 | State persists across runs | Run, exit, run again, verify state |

### Testing Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py

# Verify environment
python -c "from src.utils.config import load_env; load_env(); print('OK')"

# Test resume parsing
python -c "from src.tools.resume_tools import extract_resume_text; print(extract_resume_text('data/resume/my_resume.pdf'))"

# Test job search (real API)
python -c "from src.tools.job_tools import search_jobs; print(search_jobs('Python Developer', 'Remote'))"

# Run full workflow test
python -c "
from src.workflow.job_search_graph import run_job_search
result = run_job_search({
    'target_roles': ['Software Engineer'],
    'target_locations': ['Remote']
})
print(result)
"

# Run application tracking test
python -c "from src.tools.io_tools import add_application; print(add_application({'job_id': 'test', 'status': 'applied'}))"
```

### CLI User Flow

```
$ python src/main.py

╔══════════════════════════════════════════════════════════╗
║           JOB SEARCH ASSISTANT                           ║
║           Multi-Agent Job Search System                  ║
╚══════════════════════════════════════════════════════════╝

1. Upload Resume (PDF or DOCX)
2. Set Job Preferences
3. Search for Jobs
4. Score Resume Against Job
5. Generate Tailored Resume
6. Fill Application Form
7. Generate Cover Letter
8. Interview Preparation
0. Exit

> Enter option: 1

📄 Resume Upload
Enter path to resume file: data/resume/my_cv.pdf
✅ Resume uploaded and parsed successfully (847 words)

> Enter option: 2

💼 Job Preferences
Target roles (comma-separated): Product Manager, Senior Developer
Target locations (comma-separated): Remote, New York

✅ Preferences saved

> Enter option: 3

🔍 Searching for jobs...
Found 15 jobs from Jooble
✅ Jobs saved to state
```

### Output Files

| Type | Location | Format |
|------|----------|--------|
| Original Resume | `data/resume/` | PDF or DOCX (user provided) |
| Tailored Resumes | `data/output/resumes/` | DOCX |
| Cover Letters | `data/output/cover_letters/` | DOCX |
| Job Listings | `data/jobs.json` | JSON |

---

## Appendix: Supervisor Coordination Logic

### How Supervisor Coordinates Independent Agents

```python
# Supervisor's role is NOT a pipeline router
# Supervisor makes decisions about:

1. PRIORITIZATION
   - "Which jobs should we apply to first?"
   - "Which applications need follow-up?"

2. AGGREGATION
   - "All 5 agents found different info about Company X"
   - "The supervisor synthesizes into one view"

3. DECISION-MAKING
   - "Should we apply to this job? Score: 0.85, yes"
   - "Should we prioritize interview prep over new applications?"

4. STATE QUERY
   - Supervisor queries agents about their current status
   - Agents respond with outputs from their work
```

### Agent Independence vs Pipeline

| Pipeline Approach (Content Creation Studio) | Independent Approach (Job Search Assistant) |
|---------------------------------------------|---------------------------------------------|
| Research → Writer → Editor | Supervisor queries all agents |
| Each step waits for previous | Agents work in parallel |
| Supervisor is just a router | Supervisor makes real decisions |
| Linear flow | Coordinated but independent work |

---

*Document Version: 1.0*  
*Last Updated: 2026-05-06*  
*Purpose: Persistent memory for AI agents working on this project*