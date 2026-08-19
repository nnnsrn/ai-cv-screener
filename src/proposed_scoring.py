import json
import re
from typing import Dict, Any, Union, Tuple
import ollama

# Degree Hierarchy mapping to relative qualification levels
DEGREE_HIERARCHY = {
    "high school": 0.3,
    "associate": 0.5,
    "bachelor": 0.7,
    "master": 0.9,
    "phd": 1.0,
    "doctorate": 1.0
}


def parse_experience_years(duration_str: str) -> float:
    """
    Utility function to parse numerical years from duration strings like '3 years', '5+ yrs'.
    """
    match = re.search(r"(\d+(?:\.\d+)?)", duration_str)
    if match:
        return float(match.group(1))
    return 1.0


def score_quantitative(parsed_data: Dict[str, Any], reqs: Dict[str, Any]) -> float:
    """
    Rule-based quantitative scoring.
    Evaluates:
      1. GPA threshold against reqs.min_gpa (GPA >= min_gpa -> 1.0, below -> proportional score)
      2. Degree hierarchy against reqs.min_degree (Higher/equal degree -> 1.0, lower -> proportional score)
      3. Total experience years against reqs.min_experience_years (if specified)
      
    Parameters:
        parsed_data (dict): Structured candidate CV data.
        reqs (dict): Job requirements dictionary.
        
    Returns:
        float: Deterministic quantitative score bounded between 0.0 and 1.0.
    """
    min_gpa = float(reqs.get("min_gpa", 3.5))
    target_degree = str(reqs.get("min_degree", "Bachelor")).lower()
    min_exp_years = float(reqs.get("min_experience_years", 3.0))

    educations = parsed_data.get("educations", [])
    experiences = parsed_data.get("experiences", [])

    # 1. GPA Evaluation
    if educations:
        gpa_scores = []
        for edu in educations:
            cand_gpa = float(edu.get("gpa", 3.0))
            if cand_gpa >= min_gpa:
                gpa_score = 1.0
            else:
                gpa_score = max(0.0, cand_gpa / min_gpa)
            gpa_scores.append(gpa_score)
        avg_gpa_score = sum(gpa_scores) / len(gpa_scores)
    else:
        avg_gpa_score = 0.5

    # 2. Degree Hierarchy Evaluation
    if educations:
        degree_scores = []
        req_level = DEGREE_HIERARCHY.get(target_degree, 0.7)
        for edu in educations:
            cand_degree_raw = str(edu.get("degree", "")).lower()
            matched_level = 0.5
            for deg_name, level_score in DEGREE_HIERARCHY.items():
                if deg_name in cand_degree_raw:
                    matched_level = level_score
                    break
            # Higher degrees satisfy lower requirement level completely (>= req_level -> 1.0)
            degree_score = 1.0 if matched_level >= req_level else (matched_level / req_level)
            degree_scores.append(degree_score)
        avg_degree_score = sum(degree_scores) / len(degree_scores)
    else:
        avg_degree_score = 0.5

    # 3. Experience Duration Evaluation
    if experiences:
        total_cand_years = sum([parse_experience_years(exp.get("duration", "1 year")) for exp in experiences])
        exp_score = 1.0 if total_cand_years >= min_exp_years else max(0.0, total_cand_years / min_exp_years)
    else:
        exp_score = 0.5

    # Combine weighted quantitative components (GPA: 35%, Degree: 35%, Experience: 30%)
    quant_score = (0.35 * avg_gpa_score) + (0.35 * avg_degree_score) + (0.30 * exp_score)
    return round(float(quant_score), 4)


def score_qualitative(
    parsed_data: Dict[str, Any],
    reqs: Dict[str, Any],
    model: str = "qwen2.5:7b-instruct",
    return_details: bool = False
) -> Union[float, Tuple[float, str]]:
    """
    Evaluates candidate's experience descriptions against job requirements using Ollama qwen2.5:7b-instruct.
    Evaluates evidence present in candidate's experience without hallucinating qualifications.
    
    Parameters:
        parsed_data (dict): Structured candidate CV dict.
        reqs (dict): Job requirements dict containing description and skills.
        model (str): Ollama model name.
        return_details (bool): If True, returns tuple of (score, reasoning).
        
    Returns:
        float or Tuple[float, str]: Qualitative match score (0.0 to 1.0) or (score, reasoning).
    """
    experiences = parsed_data.get("experiences", [])
    job_description = reqs.get("description", "Senior AI/ML Engineer role.")
    req_skills = reqs.get("skills", ["Python", "PyTorch", "Machine Learning"])

    exp_text = "\n".join([
        f"- Title: {e.get('title', '')}, Duration: {e.get('duration', '')}, Description: {e.get('description', '')}"
        for e in experiences
    ])

    prompt = f"""
Evaluate the candidate's work experience strictly against the target job requirements.
Base your evaluation ONLY on concrete evidence provided in the experience descriptions below. Do not invent or assume missing qualifications.

Job Description:
{job_description}

Required Skills:
{', '.join(req_skills)}

Candidate Work Experiences:
{exp_text}

Rate the qualitative experience match on a strict scale from 0.0 (completely unfit) to 1.0 (outstanding fit).
Return ONLY valid JSON format:
{{
  "score": 0.85,
  "reasoning": "Candidate has solid experience in PyTorch NLP and microservice deployment matching role requirements."
}}
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        content = response["message"]["content"]
        data = json.loads(content)
        score = round(max(0.0, min(1.0, float(data.get("score", 0.7)))), 4)
        reasoning = str(data.get("reasoning", "Qualitative LLM match analysis completed."))
        if return_details:
            return score, reasoning
        return score
    except Exception as e:
        print(f"Notice: Ollama qualitative evaluation unavailable ({e}). Applying evidence-based fallback score.")
        match_count = sum(1 for skill in req_skills if skill.lower() in exp_text.lower())
        fallback_score = round(min(1.0, 0.4 + 0.6 * (match_count / max(1, len(req_skills)))), 4)
        fallback_reasoning = f"Rule-based fallback: matched {match_count}/{len(req_skills)} required skills in experience text."
        if return_details:
            return fallback_score, fallback_reasoning
        return fallback_score


def get_structurized_score(
    quant_score: float,
    qual_score: float,
    quant_weight: float = 0.4,
    qual_weight: float = 0.6
) -> float:
    """
    Aggregates quantitative and qualitative scores into a single Structurized Score (0.0 to 1.0).
    
    Parameters:
        quant_score (float): Rule-based quantitative score.
        qual_score (float): Qualitative LLM experience evaluation score.
        quant_weight (float): Weight for quantitative score.
        qual_weight (float): Weight for qualitative score.
        
    Returns:
        float: Combined structurized score bounded between 0.0 and 1.0.
    """
    total_w = quant_weight + qual_weight
    if total_w > 0:
        w_quant = quant_weight / total_w
        w_qual = qual_weight / total_w
    else:
        w_quant, w_qual = 0.5, 0.5

    structurized = (w_quant * quant_score) + (w_qual * qual_score)
    return round(float(structurized), 4)
