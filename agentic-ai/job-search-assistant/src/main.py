"""Job Search Assistant - CLI Main Entry Point.

A multi-agent CLI system that autonomously manages a user's job search campaign.
Coordinates independent subordinate agents to find jobs, tailor application materials,
track applications, and prepare for interviews.
"""

import sys
import logging
from pathlib import Path

# Suppress httpx and httpcore logging to keep output clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.state.job_search_state import JobSearchState, create_initial_state, Job
from src.tools.resume_tools import extract_resume_text, validate_resume_file, get_file_format
from src.tools.io_tools import (
    print_header, print_subheader, print_success, print_error, print_warning,
    print_info, print_job, print_jobs, print_score, print_menu,
    get_input, get_yes_no, confirm_action, load_state, save_state,
    print_interview_questions, clear_screen, print_job_descriptions
)
from src.tools.job_description_tools import (
    list_job_descriptions, read_job_description, save_job_description,
    search_job_descriptions, create_job_from_text, get_job_description_by_keyword,
    upload_job_description_file, validate_job_description_file
)
from src.agents.supervisor import SupervisorAgent
from src.agents.resume_tailor import ResumeTailorAgent
from src.agents.cover_letter import CoverLetterAgent
from src.agents.job_aggregator import JobAggregatorAgent
from src.agents.resume_scorer import ResumeScorerAgent
from src.agents.application_form import ApplicationFormAgent
from src.agents.interview_coach import InterviewCoachAgent
from src.conversation import (
    IntentClassifier, Intent, ConversationContext, ConversationalAgent
)
from src.workflow.job_search_graph import JobSearchGraph
from src.utils.config import load_env


# State file path
STATE_FILE = project_root / "data" / "state.json"


class JobSearchAssistant:
    """Main CLI application for Job Search Assistant."""
    
    def __init__(self):
        """Initialize the application."""
        load_env()
        self.state: JobSearchState = self._load_or_create_state()
        self._initialize_agents()
    
    def _load_or_create_state(self) -> JobSearchState:
        """Load existing state or create new one."""
        if STATE_FILE.exists():
            state_data = load_state(str(STATE_FILE))
            if state_data:
                # Convert to JobSearchState
                return JobSearchState(**state_data)
        return create_initial_state()
    
    def _save_state(self) -> None:
        """Save current state to file."""
        save_state(dict(self.state), str(STATE_FILE))
    
    def _initialize_agents(self) -> None:
        """Initialize all agents."""
        try:
            self.supervisor = SupervisorAgent()
            self.resume_tailor = ResumeTailorAgent()
            self.cover_letter = CoverLetterAgent()
            self.job_aggregator = JobAggregatorAgent()
            self.resume_scorer = ResumeScorerAgent()
            self.application_form = ApplicationFormAgent()
            self.interview_coach = InterviewCoachAgent()
            self.graph = JobSearchGraph()
        except ValueError as e:
            print_error(str(e))
            print_info("Please ensure your .env file is set up with API keys.")
            sys.exit(1)
    
    def run(self) -> None:
        """Run the main application loop."""
        while True:
            clear_screen()
            print_header("JOB SEARCH ASSISTANT")
            
            # Show current status
            self._show_status()
            
            # Show menu
            options = [
                ("1", "Upload Resume (PDF or DOCX)"),
                ("2", "Set Job Preferences"),
                ("3", "Search for Jobs"),
                ("4", "Score Resume Against Job"),
                ("5", "Generate Tailored Resume"),
                ("6", "Fill Application Form"),
                ("7", "Generate Cover Letter"),
                ("8", "Interview Preparation"),
                ("9", "View Saved Jobs"),
                ("A", "Add Job Description"),
                ("B", "Manage Local Job Descriptions"),
                ("C", "Upload Job Description File"),
                ("T", "Chat Mode (Natural Language)"),
                ("0", "Exit"),
            ]
            print_menu(options)
            
            choice = get_input("Enter option", required=True)
            
            if choice == "0":
                self._exit()
                break
            elif choice == "1":
                self._upload_resume()
            elif choice == "2":
                self._set_preferences()
            elif choice == "3":
                self._search_jobs()
            elif choice == "4":
                self._score_resume()
            elif choice == "5":
                self._tailor_resume()
            elif choice == "6":
                self._fill_form()
            elif choice == "7":
                self._generate_cover_letter()
            elif choice == "8":
                self._interview_prep()
            elif choice == "9":
                self._view_jobs()
            elif choice.lower() == "a":
                self._add_job_description()
            elif choice.lower() == "b":
                self._manage_job_descriptions()
            elif choice.lower() == "c":
                self._upload_job_description_file()
            elif choice.lower() == "t":
                self._run_chat_mode()
            else:
                print_warning("Invalid option. Please try again.")
            
            input("\nPress Enter to continue...")
    
    def _show_status(self) -> None:
        """Show current status."""
        print_subheader("Current Status")
        
        # Resume status
        if self.state.get("original_resume_text"):
            words = len(self.state["original_resume_text"].split())
            print(f"  Resume: Uploaded ({words} words)")
        else:
            print("  Resume: Not uploaded")
        
        # Preferences
        roles = self.state.get("target_roles", [])
        locations = self.state.get("target_locations", [])
        if roles:
            print(f"  Target Roles: {', '.join(roles)}")
        if locations:
            print(f"  Target Locations: {', '.join(locations)}")
        
        # Jobs found
        jobs = self.state.get("job_aggregator_output", {}).get("jobs", [])
        print(f"  Jobs Found (API): {len(jobs)}")
        
        # Local job descriptions
        local_jds = list_job_descriptions()
        print(f"  Job Descriptions (Local): {len(local_jds)}")
        
        # Current phase
        phase = self.state.get("current_phase", "idle")
        print(f"  Phase: {phase}")
        
        # Blockers
        blockers = self.state.get("blockers", [])
        if blockers:
            print(f"  [!] Blockers: {', '.join(blockers)}")
    
    def _upload_resume(self) -> None:
        """Handle resume upload."""
        print_subheader("Resume Upload")
        
        path = get_input("Enter path to resume file (PDF or DOCX)")
        
        # Validate file
        is_valid, error = validate_resume_file(path)
        if not is_valid:
            print_error(error)
            return
        
        try:
            # Extract text
            text = extract_resume_text(path)
            words = len(text.split())
            
            # Update state
            self.state["original_resume_path"] = path
            self.state["original_resume_text"] = text
            
            print_success(f"Resume uploaded and parsed successfully ({words} words)")
            self._save_state()
            
        except Exception as e:
            print_error(f"Failed to parse resume: {e}")
    
    def _set_preferences(self) -> None:
        """Set job search preferences."""
        print_subheader("Job Preferences")
        
        roles_input = get_input("Target roles (comma-separated)")
        roles = [r.strip() for r in roles_input.split(",") if r.strip()]
        
        locations_input = get_input("Target locations (comma-separated)")
        locations = [l.strip() for l in locations_input.split(",") if l.strip()]
        
        companies_input = get_input("Target companies (comma-separated, optional)")
        companies = [c.strip() for c in companies_input.split(",") if c.strip()]
        
        # Update state
        self.state["target_roles"] = roles
        self.state["target_locations"] = locations
        self.state["target_companies"] = companies
        
        # Also update user profile
        self.state["user_profile"]["target_roles"] = roles
        self.state["user_profile"]["target_locations"] = locations
        self.state["user_profile"]["target_companies"] = companies
        
        print_success("Preferences saved")
        self._save_state()
    
    def _search_jobs(self) -> None:
        """Search for jobs using Jooble API."""
        print_subheader("Job Search")
        
        roles = self.state.get("target_roles", [])
        locations = self.state.get("target_locations", [])
        
        if not roles:
            print_error("No target roles set. Please set preferences first.")
            return
        
        print_info("Searching for jobs via Jooble API...")
        
        try:
            result = self.job_aggregator.search_for_jobs(self.state)
            
            if result.get("success"):
                jobs = result.get("jobs", [])
                print_success(f"Found {len(jobs)} jobs")
                
                if jobs:
                    print("\nTop 5 jobs:")
                    for i, job in enumerate(jobs[:5], 1):
                        print_job(job, i)
            else:
                print_error(result.get("error", "Search failed"))
                
        except Exception as e:
            print_error(f"Job search failed: {e}")
        
        self._save_state()
    
    def _score_resume(self) -> None:
        """Score resume against a job."""
        print_subheader("Resume Score Check")
        
        if not self.state.get("original_resume_text"):
            print_error("No resume uploaded. Please upload a resume first.")
            return
        
        jobs = self.state.get("job_aggregator_output", {}).get("jobs", [])
        local_jds = list_job_descriptions()
        
        # Combine sources - prefer API jobs, but allow local job descriptions
        has_jobs = bool(jobs)
        has_local_jds = bool(local_jds)
        
        if not has_jobs and not has_local_jds:
            print_error("No jobs found. Please search for jobs or upload job descriptions first.")
            return
        
        # Let user choose source
        print("\nSelect scoring source:")
        print("  1. Search results (API jobs)")
        if has_local_jds:
            print(f"  2. Uploaded job descriptions ({len(local_jds)} available)")
        
        if has_jobs and has_local_jds:
            source_choice = get_input("Enter option (1-2)")
        elif has_local_jds:
            source_choice = "2"
        else:
            source_choice = "1"
        
        job_description = ""
        job_id = ""
        
        if source_choice == "2" and has_local_jds:
            # Use local job descriptions
            print("\nSelect a job description:")
            for i, jd in enumerate(local_jds[:5], 1):
                print(f"  {i}. {jd.get('title', 'N/A')} at {jd.get('company', 'N/A')}")
            
            choice = get_input("Enter job description number (1-5)")
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < min(5, len(local_jds)):
                    jd = local_jds[idx]
                    job_description = jd.get("description", "")
                    job_id = jd.get("id", "local")
                    # Also set the selected job description ID in state
                    self.state["selected_job_description_id"] = jd.get("id", "")
            except ValueError:
                print_error("Please enter a valid number")
                return
        else:
            # Use API search results
            if not has_jobs:
                print_error("No search results available. Please search for jobs first.")
                return
            
            print("\nSelect a job to score against:")
            for i, job in enumerate(jobs[:5], 1):
                print(f"  {i}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
            
            choice = get_input("Enter job number (1-5)")
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < min(5, len(jobs)):
                    job = jobs[idx]
                    job_description = job.get("description", "")
                    job_id = job.get("id", "")
                else:
                    print_error("Invalid selection")
                    return
            except ValueError:
                print_error("Please enter a valid number")
                return
        
        print_info("Scoring resume...")
        result = self.resume_scorer.score_resume(
            self.state, job_description, job_id
        )
        
        if result.get("success"):
            print_score(result)
        else:
            print_error(result.get("error", "Scoring failed"))
        
        self._save_state()
    
    def _tailor_resume(self) -> None:
        """Generate tailored resume for a job."""
        print_subheader("Tailor Resume")
        
        if not self.state.get("original_resume_text"):
            print_error("No resume uploaded. Please upload a resume first.")
            return
        
        jobs = self.state.get("job_aggregator_output", {}).get("jobs", [])
        local_jds = list_job_descriptions()
        
        # Combine sources
        has_jobs = bool(jobs)
        has_local_jds = bool(local_jds)
        
        if not has_jobs and not has_local_jds:
            print_error("No jobs found. Please search for jobs or upload job descriptions first.")
            return
        
        # Let user choose source
        print("\nSelect source:")
        print("  1. Search results (API jobs)")
        if has_local_jds:
            print(f"  2. Uploaded job descriptions ({len(local_jds)} available)")
        
        if has_jobs and has_local_jds:
            source_choice = get_input("Enter option (1-2)")
        elif has_local_jds:
            source_choice = "2"
        else:
            source_choice = "1"
        
        job_description = ""
        job_id = ""
        
        if source_choice == "2" and has_local_jds:
            # Use local job descriptions
            print("\nSelect a job description:")
            for i, jd in enumerate(local_jds[:5], 1):
                print(f"  {i}. {jd.get('title', 'N/A')} at {jd.get('company', 'N/A')}")
            
            choice = get_input("Enter job description number (1-5)")
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < min(5, len(local_jds)):
                    jd = local_jds[idx]
                    job_description = jd.get("description", "")
                    job_id = jd.get("id", "local")
                    self.state["selected_job_description_id"] = jd.get("id", "")
                else:
                    print_error("Invalid selection")
                    return
            except ValueError:
                print_error("Please enter a valid number")
                return
        else:
            # Use API search results
            if not has_jobs:
                print_error("No search results available. Please search for jobs first.")
                return
            
            print("\nSelect a job to tailor resume for:")
            for i, job in enumerate(jobs[:5], 1):
                print(f"  {i}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
            
            choice = get_input("Enter job number (1-5)")
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < min(5, len(jobs)):
                    job = jobs[idx]
                    job_description = job.get("description", "")
                    job_id = job.get("id", "")
                else:
                    print_error("Invalid selection")
                    return
            except ValueError:
                print_error("Please enter a valid number")
                return
        
        # Choose operation
        print("\nSelect tailoring operation:")
        print("  1. Full Tailor (rewrite entire resume)")
        print("  2. Keyword Inject (add missing keywords)")
        print("  3. Section Edit (modify specific section)")
        
        op_choice = get_input("Enter option (1-3)")
        operation_map = {"1": "full_tailor", "2": "keyword_inject", "3": "section_edit"}
        operation = operation_map.get(op_choice, "full_tailor")
        
        print_info("Generating tailored resume...")
        result = self.resume_tailor.tailor_resume(
            self.state, job_description, job_id, operation
        )
        
        if result.get("success"):
            path = result.get("tailored_resume_docx_path", "")
            print_success(f"Tailored resume saved to: {path}")
        else:
            print_error(result.get("error", "Tailoring failed"))
        
        self._save_state()
    
    def _fill_form(self) -> None:
        """Fill application form fields."""
        print_subheader("Application Form Assistance")
        
        if not self.state.get("original_resume_text"):
            print_error("No resume uploaded. Please upload a resume first.")
            return
        
        # Get field info
        platform = get_input("Platform (e.g., LinkedIn, Greenhouse, generic)")
        
        fields_input = get_input("Fields to fill (comma-separated, e.g., 'Years of Experience, Current Title, Skills')")
        field_names = [f.strip() for f in fields_input.split(",") if f.strip()]
        
        fields = [{"name": name, "type": "text"} for name in field_names]
        
        print_info("Generating field suggestions...")
        result = self.application_form.fill_form_fields(
            self.state, fields, platform
        )
        
        if result.get("success"):
            print_success("Field suggestions generated:")
            answers = result.get("field_answers", {})
            for field, answer in answers.items():
                print(f"\n  {field}:")
                print(f"    {answer}")
        else:
            print_error("Form filling failed")
        
        self._save_state()
    
    def _generate_cover_letter(self) -> None:
        """Generate cover letter for a job."""
        print_subheader("Cover Letter Generation")
        
        if not self.state.get("original_resume_text"):
            print_error("No resume uploaded. Please upload a resume first.")
            return
        
        jobs = self.state.get("job_aggregator_output", {}).get("jobs", [])
        if not jobs:
            print_error("No jobs found. Please search for jobs first.")
            return
        
        # Show top jobs
        print("Select a job for the cover letter:")
        for i, job in enumerate(jobs[:5], 1):
            print(f"  {i}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
        
        choice = get_input("Enter job number (1-5)")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < min(5, len(jobs)):
                job = jobs[idx]
                
                save_docx = get_yes_no("Save as DOCX?")
                
                print_info("Generating cover letter...")
                result = self.cover_letter.generate_cover_letter(
                    self.state, job, save_docx
                )
                
                if result.get("success"):
                    print_success("Cover letter generated:")
                    print("\n" + result.get("cover_letter_text", ""))
                    
                    if result.get("cover_letter_docx_path"):
                        print(f"\nSaved to: {result.get('cover_letter_docx_path')}")
                else:
                    print_error(result.get("error", "Generation failed"))
            else:
                print_error("Invalid selection")
                
        except ValueError:
            print_error("Please enter a valid number")
        
        self._save_state()
    
    def _interview_prep(self) -> None:
        """Prepare for interview."""
        print_subheader("Interview Preparation")
        
        jobs = self.state.get("job_aggregator_output", {}).get("jobs", [])
        
        if not jobs:
            print_info("No jobs available. Preparing general interview materials.")
            job_description = ""
            company = ""
        else:
            print("Select a job for interview prep:")
            for i, job in enumerate(jobs[:3], 1):
                print(f"  {i}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
            
            choice = get_input("Enter job number (1-3), or 0 for general prep")
            
            if choice == "0":
                job_description = ""
                company = ""
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < min(3, len(jobs)):
                        job = jobs[idx]
                        job_description = job.get("description", "")
                        company = job.get("company", "")
                    else:
                        job_description = ""
                        company = ""
                except ValueError:
                    job_description = ""
                    company = ""
        
        print_info("Generating interview materials...")
        result = self.interview_coach.prepare_interview(
            self.state, job_description, company
        )
        
        if result.get("success"):
            print_success("Interview materials generated:")
            print("\n" + result.get("materials", ""))
        else:
            print_error("Interview prep failed")
        
        self._save_state()
    
    def _view_jobs(self) -> None:
        """View saved jobs."""
        print_subheader("Saved Jobs")
        
        jobs = self.state.get("job_aggregator_output", {}).get("jobs", [])
        
        if not jobs:
            print_info("No jobs saved. Please search for jobs first.")
            return
        
        print_jobs(jobs, limit=10)
    
    def _add_job_description(self) -> None:
        """Add a new job description."""
        print_subheader("Add Job Description")
        
        title = get_input("Job Title")
        company = get_input("Company Name")
        location = get_input("Location")
        
        print("\nEnter the full job description (press Enter twice to finish):")
        print("-" * 40)
        
        lines = []
        while True:
            line = input()
            if line == "" and (not lines or lines[-1] == ""):
                break
            lines.append(line)
        
        description = "\n".join(lines)
        
        requirements_input = get_input("Requirements (comma-separated, optional)")
        requirements = [r.strip() for r in requirements_input.split(",") if r.strip()]
        
        try:
            path = create_job_from_text(
                title=title,
                company=company,
                location=location,
                description=description,
                requirements=requirements if requirements else None
            )
            print_success(f"Job description saved to: {path}")
            
            # Update state
            self.state["local_job_descriptions"] = list_job_descriptions()
            self._save_state()
            
        except Exception as e:
            print_error(f"Failed to save job description: {e}")
    
    def _manage_job_descriptions(self) -> None:
        """Manage local job descriptions."""
        print_subheader("Local Job Descriptions")
        
        job_descriptions = list_job_descriptions()
        
        if not job_descriptions:
            print_info("No job descriptions found in data/job-descriptions/")
            print_info("You can add job descriptions using option 'A' from the main menu.")
            return
        
        print_job_descriptions(job_descriptions)
        
        print("\nOptions:")
        print("  1. View full description")
        print("  2. Search descriptions")
        print("  3. Use for resume tailoring")
        print("  4. Use for cover letter")
        print("  0. Back")
        
        choice = get_input("Enter option")
        
        if choice == "1":
            self._view_job_description_details(job_descriptions)
        elif choice == "2":
            self._search_job_descriptions()
        elif choice == "3":
            self._use_job_description_for_tailoring(job_descriptions)
        elif choice == "4":
            self._use_job_description_for_cover_letter(job_descriptions)
    
    def _upload_job_description_file(self) -> None:
        """Upload a job description from a file (PDF, DOCX, TXT, or MD)."""
        print_subheader("Upload Job Description File")
        
        file_path = get_input("Enter path to job description file (PDF, DOCX, TXT, or MD)")
        
        # Validate file
        is_valid, error = validate_job_description_file(file_path)
        if not is_valid:
            print_error(error)
            return
        
        try:
            # Optional metadata
            title = get_input("Job title (press Enter to auto-detect from file)")
            title = title.strip() if title.strip() else None
            
            company = get_input("Company name (optional)")
            company = company.strip() if company.strip() else None
            
            location = get_input("Job location (optional)")
            location = location.strip() if location.strip() else None
            
            # Upload and parse
            result = upload_job_description_file(
                file_path,
                title=title,
                company=company,
                location=location
            )
            
            print_success(f"Job description uploaded successfully!")
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  Company: {result.get('company', 'N/A')}")
            print(f"  Location: {result.get('location', 'N/A')}")
            print(f"  Saved to: {result.get('path', 'N/A')}")
            
            # Refresh state
            self.state["local_job_descriptions"] = list_job_descriptions()
            self._save_state()
            
        except Exception as e:
            print_error(f"Failed to upload job description: {e}")
    
    def _view_job_description_details(self, job_descriptions: list) -> None:
        """View full details of a job description."""
        choice = get_input("Enter job description number to view")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(job_descriptions):
                jd_id = job_descriptions[idx]["id"]
                jd = read_job_description(jd_id)
                
                if jd:
                    print_subheader(f"Job Description: {jd.get('title', 'N/A')}")
                    print(f"\nCompany: {jd.get('company', 'N/A')}")
                    print(f"Location: {jd.get('location', 'N/A')}")
                    print(f"\nDescription:")
                    print("-" * 40)
                    print(jd.get("description", ""))
                    
                    raw_data = jd.get("raw_data", {})
                    if raw_data.get("requirements"):
                        print(f"\nRequirements:")
                        for req in raw_data["requirements"]:
                            print(f"  - {req}")
            else:
                print_error("Invalid selection")
        except ValueError:
            print_error("Please enter a valid number")
    
    def _search_job_descriptions(self) -> None:
        """Search job descriptions by keyword."""
        query = get_input("Enter search keyword")
        
        results = search_job_descriptions(query)
        
        if not results:
            print_info("No matching job descriptions found.")
            return
        
        print(f"\nFound {len(results)} matching job descriptions:")
        print_job_descriptions(results)
    
    def _use_job_description_for_tailoring(self, job_descriptions: list) -> None:
        """Use a local job description for resume tailoring."""
        if not self.state.get("original_resume_text"):
            print_error("No resume uploaded. Please upload a resume first.")
            return
        
        choice = get_input("Enter job description number to use")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(job_descriptions):
                jd_id = job_descriptions[idx]["id"]
                jd = read_job_description(jd_id)
                
                if jd:
                    print_info("Generating tailored resume...")
                    result = self.resume_tailor.tailor_resume(
                        self.state,
                        jd.get("description", ""),
                        jd_id
                    )
                    
                    if result.get("success"):
                        print_success(f"Tailored resume saved to: {result.get('tailored_resume_docx_path')}")
                    else:
                        print_error(result.get("error", "Tailoring failed"))
            else:
                print_error("Invalid selection")
        except ValueError:
            print_error("Please enter a valid number")
    
    def _use_job_description_for_cover_letter(self, job_descriptions: list) -> None:
        """Use a local job description for cover letter."""
        if not self.state.get("original_resume_text"):
            print_error("No resume uploaded. Please upload a resume first.")
            return
        
        choice = get_input("Enter job description number to use")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(job_descriptions):
                jd_id = job_descriptions[idx]["id"]
                jd = read_job_description(jd_id)
                
                if jd:
                    # Convert to job dict format
                    job = {
                        "id": jd_id,
                        "title": jd.get("title", ""),
                        "company": jd.get("company", ""),
                        "description": jd.get("description", ""),
                        "location": jd.get("location", ""),
                    }
                    
                    save_docx = get_yes_no("Save cover letter as DOCX?")
                    
                    print_info("Generating cover letter...")
                    result = self.cover_letter.generate_cover_letter(
                        self.state, job, save_docx
                    )
                    
                    if result.get("success"):
                        print_success("Cover letter generated:")
                        print("\n" + result.get("cover_letter_text", "")[:500] + "...")
                    else:
                        print_error(result.get("error", "Generation failed"))
            else:
                print_error("Invalid selection")
        except ValueError:
            print_error("Please enter a valid number")
    
    def _run_chat_mode(self) -> None:
        """Run the conversational (natural language) interface."""
        clear_screen()
        print_header("CHAT MODE - Natural Language Interface")
        print_info("Type your requests naturally. Type 'exit' to return to menu.")
        print()
        
        # Initialize the conversational agent
        chat_agent = ConversationalAgent(self.state)
        
        # Show greeting
        print("Bot: " + chat_agent.process_message("hello"))
        print()
        
        while True:
            try:
                user_input = get_input("You", required=True)
                
                if user_input.lower() in ["exit", "quit", "back", "menu"]:
                    print_info("Exiting chat mode...")
                    # Save any state changes
                    self._save_state()
                    break
                
                # Process the message
                response = chat_agent.process_message(user_input)
                print(f"\nBot: {response}\n")
                
                # Check if conversation ended (e.g., user said goodbye)
                if "goodbye" in response.lower():
                    input("Press Enter to return to menu...")
                    break
                    
            except KeyboardInterrupt:
                print_info("\nExiting chat mode...")
                self._save_state()
                break
            except Exception as e:
                print_error(f"Error in chat mode: {e}")
    
    def _exit(self) -> None:
        """Exit the application."""
        self._save_state()
        print_info("State saved. Goodbye!")


def main():
    """Main entry point."""
    try:
        app = JobSearchAssistant()
        app.run()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print_error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()