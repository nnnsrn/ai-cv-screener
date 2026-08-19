import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import ollama

# Pydantic Schemas for Structured CV parsing validation
class EducationItem(BaseModel):
    degree: str = Field(default="Bachelor", description="Degree title e.g. Bachelor, Master, PhD")
    gpa: float = Field(default=3.0, description="Grade point average e.g. 3.5")

class ExperienceItem(BaseModel):
    title: str = Field(default="", description="Job title")
    duration: str = Field(default="", description="Duration e.g. 3 years")
    description: str = Field(default="", description="Job responsibilities and details")

class StructuredCV(BaseModel):
    educations: List[EducationItem] = Field(default_factory=list)
    experiences: List[ExperienceItem] = Field(default_factory=list)
    etc: Dict[str, Any] = Field(default_factory=lambda: {"certifications": []})

class LegacyFeatures(BaseModel):
    summary: str = Field(default="")
    skills: List[str] = Field(default_factory=list)


def extract_legacy_features(raw_text: str, model: str = "qwen2.5:7b-instruct") -> Dict[str, Any]:
    """
    Legacy CV parsing method using Ollama qwen2.5:7b-instruct.
    Extracts a concise CV summary and a list of key skills.
    
    Parameters:
        raw_text (str): Extracted CV text.
        model (str): Ollama model name.
        
    Returns:
        Dict[str, Any]: {"summary": str, "skills": List[str]}
    """
    prompt = f"""
Given the following CV text, extract a concise overall summary of the candidate's experience and a list of key technical/professional skills.
Return ONLY valid JSON with this exact schema:
{{
  "summary": "Concise candidate summary",
  "skills": ["Skill 1", "Skill 2"]
}}

CV Text:
{raw_text}
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        content = response["message"]["content"]
        data = json.loads(content)
        return {
            "summary": str(data.get("summary", "")).strip(),
            "skills": [str(s).strip() for s in data.get("skills", []) if s]
        }
    except Exception as e:
        print(f"Notice: Ollama legacy parsing unavailable ({e}). Applying lightweight fallback.")
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        summary = " ".join(lines[:3]) if lines else "Candidate CV Summary"
        skills = [w for w in ["Python", "Machine Learning", "PyTorch", "NLP", "Docker", "SQL", "FastAPI"] if w.lower() in raw_text.lower()]
        return {"summary": summary, "skills": skills}


def extract_structured_data(raw_text: str, model: str = "qwen2.5:7b-instruct") -> Dict[str, Any]:
    """
    Proposed Structured CV parsing method using Ollama qwen2.5:7b-instruct.
    Requires structured JSON matching schema (educations, experiences, etc.certifications)
    and validates using Pydantic.
    
    Parameters:
        raw_text (str): Extracted CV text.
        model (str): Ollama model identifier.
        
    Returns:
        Dict[str, Any]: Validated structured CV dictionary.
    """
    prompt = f"""
Extract structured components from this CV text.
Return ONLY valid JSON matching this exact structure:
{{
  "educations": [
    {{"degree": "Master of Science in Computer Science", "gpa": 3.8}}
  ],
  "experiences": [
    {{"title": "Software Engineer", "duration": "3 years", "description": "Built PyTorch models and microservices."}}
  ],
  "etc": {{
    "certifications": ["AWS Certified Cloud Practitioner"]
  }}
}}

CV Text:
{raw_text}
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        content = response["message"]["content"]
        data = json.loads(content)
        validated = StructuredCV(**data)
        return validated.model_dump()
    except Exception as e:
        print(f"Notice: Ollama structured parsing model error or unavailable ({e}). Using schema-aligned extraction fallback.")
        # Rule-based structured extraction for offline/mock environments
        gpa_match = re.search(r"GPA[:\s]+([0-3]\.\d+|4\.0)", raw_text, re.IGNORECASE)
        gpa_val = float(gpa_match.group(1)) if gpa_match else 3.2

        degree_val = "Bachelor"
        if "master" in raw_text.lower() or "m.s." in raw_text.lower():
            degree_val = "Master"
        elif "phd" in raw_text.lower() or "doctor" in raw_text.lower():
            degree_val = "PhD"

        dur_match = re.search(r"(\d+)\s+years?", raw_text, re.IGNORECASE)
        dur_val = f"{dur_match.group(1)} years" if dur_match else "2 years"

        certs = []
        if "certified" in raw_text.lower() or "certification" in raw_text.lower():
            certs = ["Professional Certification"]

        fallback_dict = {
            "educations": [{"degree": degree_val, "gpa": gpa_val}],
            "experiences": [{"title": "Engineer / Developer", "duration": dur_val, "description": raw_text[:300]}],
            "etc": {"certifications": certs}
        }
        validated = StructuredCV(**fallback_dict)
        return validated.model_dump()
