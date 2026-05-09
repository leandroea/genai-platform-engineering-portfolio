"""Intent classifier for natural language understanding."""

from enum import Enum
from typing import Optional
import re


class Intent(Enum):
    """Enumeration of possible user intents."""
    GREETING = "greeting"
    HELP = "help"
    UPLOAD_RESUME = "upload_resume"
    SET_PREFERENCES = "set_preferences"
    SEARCH_JOBS = "search_jobs"
    SCORE_RESUME = "score_resume"
    TAILOR_RESUME = "tailor_resume"
    GENERATE_COVER_LETTER = "generate_cover_letter"
    FILL_FORM = "fill_form"
    INTERVIEW_PREP = "interview_prep"
    VIEW_JOBS = "view_jobs"
    MANAGE_JOB_DESCRIPTIONS = "manage_job_descriptions"
    UPLOAD_JOB_DESCRIPTION = "upload_job_description"
    ADD_JOB_DESCRIPTION = "add_job_description"
    GENERAL_QUERY = "general_query"
    EXIT = "exit"
    UNKNOWN = "unknown"


class IntentClassifier:
    """Classifies user input into intents using keyword matching.
    
    This is a simple rule-based classifier that uses keyword patterns
    to determine the user's intent. It can be enhanced with ML/LLM
    for better accuracy.
    """
    
    # Intent patterns - order matters (more specific first)
    INTENT_PATTERNS = {
        Intent.UPLOAD_RESUME: [
            r"upload.*resume",
            r"add.*resume",
            r"load.*resume",
            r"my.*resume",
            r"i have.*resume",
            r"resume.*upload",
            r"new.*resume",
        ],
        Intent.SET_PREFERENCES: [
            r"set.*preference",
            r"target.*(job|role|location|company)",
            r"looking.*for.*(?:jobs?|roles?)",
            r"i want.*(?:jobs?|roles?)",
            r"preference",
            r"target.*location",
            r"target.*role",
            r"remote",
            r"hybrid",
        ],
        Intent.SEARCH_JOBS: [
            r"search.*(?:for)?.*jobs?",
            r"find.*(?:me)?.*jobs?",
            r"look.*for.*jobs?",
            r"find.*jobs?",
            r"search.*jobs?",
            r"find.*positions?",
            r"job.*search",
        ],
        Intent.SCORE_RESUME: [
            r"score.*resume",
            r"check.*resume.*(?:against|for)",
            r"evaluate.*resume",
            r"how.*(?:does|is).*resume",
            r"resume.*score",
            r"match.*resume",
        ],
        Intent.TAILOR_RESUME: [
            r"tailor.*resume",
            r"customize.*resume",
            r"adjust.*resume",
            r"rewrite.*resume",
            r"modify.*resume",
            r"make.*resume.*(?:for|to)",
            r"resume.*for.*(?:specific|this|that)",
        ],
        Intent.GENERATE_COVER_LETTER: [
            r"cover.*letter",
            r"write.*letter",
            r"generate.*letter",
            r"create.*letter",
            r"letter.*for",
            r"cover.*note",
        ],
        Intent.FILL_FORM: [
            r"fill.*form",
            r"application.*form",
            r"form.*field",
            r"help.*form",
            r"complete.*form",
            r"apply.*form",
        ],
        Intent.INTERVIEW_PREP: [
            r"interview.*(?:prep|prepare|question)",
            r"prepare.*(?:for)?.*interview",
            r"practice.*interview",
            r"interview.*question",
            r"interview.*tips",
            r"mock.*interview",
        ],
        Intent.VIEW_JOBS: [
            r"view.*jobs",
            r"show.*jobs",
            r"list.*jobs",
            r"see.*jobs",
            r"display.*jobs",
            r"what.*jobs",
            r"found.*jobs",
        ],
        Intent.MANAGE_JOB_DESCRIPTIONS: [
            r"manage.*job.*description",
            r"job.*description.*list",
            r"view.*description",
            r"see.*description",
        ],
        Intent.UPLOAD_JOB_DESCRIPTION: [
            r"upload.*job.*description",
            r"add.*job.*description.*file",
            r"job.*description.*pdf",
            r"job.*description.*docx",
        ],
        Intent.ADD_JOB_DESCRIPTION: [
            r"add.*job.*description",
            r"create.*job.*description",
            r"new.*job.*description",
            r"enter.*job.*description",
        ],
        Intent.HELP: [
            r"help",
            r"what.*can.*(?:you|i).*do",
            r"how.*(?:can|do).*(?:you|i)",
            r"what.*option",
            r"assist",
            r"support",
        ],
        Intent.GREETING: [
            r"^(?:hi|hello|hey|greetings|good.*(?:morning|afternoon|evening))",
            r"howdy",
            r"sup",
        ],
        Intent.EXIT: [
            r"exit",
            r"quit",
            r"bye",
            r"goodbye",
            r"see.*you",
            r"done",
        ],
    }
    
    # Entity patterns for extracting specific information
    ENTITY_PATTERNS = {
        "location": [
            r"\b(?:in|at|near|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            r"\b(?:san francisco|nyc|new york|los angeles|seattle|austin|boston|denver|chicago)\b",
        ],
        "job_title": [
            r"\b(?:senior|junior|lead|principal|staff)\s+\w+",
            r"\bsoftware\s+engineer\b",
            r"\bdata\s+scientist\b",
            r"\bproduct\s+manager\b",
            r"\bfrontend|backend|full.?stack\b",
        ],
        "company": [
            r"\bat\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b",
            r"\bfor\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b",
            r"\bcompany\s+([A-Z][a-zA-Z]+)\b",
        ],
    }
    
    def classify(self, user_input: str) -> Intent:
        """Classify user input into an intent.
        
        Args:
            user_input: The user's natural language input
            
        Returns:
            The classified Intent
        """
        if not user_input or not user_input.strip():
            return Intent.UNKNOWN
            
        input_lower = user_input.lower().strip()
        
        # Check each intent pattern in order
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, input_lower, re.IGNORECASE):
                    return intent
        
        return Intent.GENERAL_QUERY
    
    def extract_entities(self, user_input: str) -> dict:
        """Extract entities from user input.
        
        Args:
            user_input: The user's natural language input
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        input_lower = user_input.lower()
        
        # Extract locations
        locations = []
        for pattern in self.ENTITY_PATTERNS["location"]:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            locations.extend(matches)
        if locations:
            entities["locations"] = list(set(locations))
        
        # Extract job titles
        job_titles = []
        for pattern in self.ENTITY_PATTERNS["job_title"]:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            job_titles.extend(matches)
        if job_titles:
            entities["job_titles"] = list(set(job_titles))
        
        # Extract companies
        companies = []
        for pattern in self.ENTITY_PATTERNS["company"]:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            companies.extend(matches)
        if companies:
            entities["companies"] = list(set(companies))
        
        return entities
    
    def classify_with_context(self, user_input: str, context: dict) -> tuple[Intent, dict]:
        """Classify intent while considering conversation context.
        
        This method uses previous context to help disambiguate
        vague commands.
        
        Args:
            user_input: The user's natural language input
            context: Current conversation context
            
        Returns:
            Tuple of (Intent, entities dict)
        """
        intent = self.classify(user_input)
        entities = self.extract_entities(user_input)
        
        # Handle context-dependent disambiguation
        if intent == Intent.GENERAL_QUERY and context.get("pending_intent"):
            # User might be responding to a follow-up question
            pending = context.get("pending_intent")
            if context.get("awaiting_confirmation"):
                # User likely confirming or denying
                if any(word in user_input.lower() for word in ["yes", "yeah", "sure", "ok", "yep"]):
                    return pending, entities
                elif any(word in user_input.lower() for word in ["no", "nope", "nah", "skip"]):
                    return Intent.UNKNOWN, entities  # User declined
        
        return intent, entities