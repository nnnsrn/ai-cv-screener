import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util

# Global singletons for lazy loading when not pre-supplied
_BGE_MODEL = None
_SPACY_NLP = None


def get_bge_model(device: str = None) -> SentenceTransformer:
    """
    Initializes and returns the singleton BAAI/bge-m3 SentenceTransformer model.
    Falls back to a lightweight model if BGE-M3 fails to load.
    """
    global _BGE_MODEL
    if _BGE_MODEL is None:
        try:
            _BGE_MODEL = SentenceTransformer("BAAI/bge-m3", device=device)
        except Exception as e:
            print(f"Notice: Loading fallback SentenceTransformer model due to: {e}")
            _BGE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _BGE_MODEL


def get_spacy_nlp() -> spacy.language.Language:
    """
    Initializes and returns the singleton spaCy en_core_web_md NLP pipeline.
    Raises clear error if en_core_web_md is missing.
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_md")
        except Exception as e:
            raise OSError(
                "spaCy model 'en_core_web_md' is not installed in this environment. "
                "Please run 'bash setup.sh' or 'python -m spacy download en_core_web_md' to install it."
            ) from e
    return _SPACY_NLP


def scale_legacy_score(
    raw_score: float,
    min_in: float = 0.3,
    max_in: float = 0.7,
    min_out: float = 0.5,
    max_out: float = 0.95
) -> float:
    """
    Legacy score scaling transformation:
    Maps raw similarity scores typically centered around [0.3, 0.7] into an intuitive baseline range [0.5, 0.95].
    Handles values outside the interval predictably via linear mapping and clipping.
    """
    if raw_score <= min_in:
        return float(min_out)
    if raw_score >= max_in:
        return float(max_out)
    scaled = min_out + (raw_score - min_in) * (max_out - min_out) / (max_in - min_in)
    return round(float(scaled), 4)


def get_semantic_score(
    job_query: str,
    cv_summary: str,
    model: SentenceTransformer = None
) -> float:
    """
    Calculates BAAI/bge-m3 cosine similarity between job query and CV summary,
    then applies legacy scaling transformation.
    
    Parameters:
        job_query (str): Target job requirements/query string.
        cv_summary (str): High-level CV summary text.
        model (SentenceTransformer, optional): Pre-initialized BGE-M3 model instance.
        
    Returns:
        float: Scaled semantic score (0.0 to 1.0).
    """
    if not job_query or not cv_summary:
        return 0.0

    if model is None:
        model = get_bge_model()

    emb_query = model.encode(job_query, convert_to_tensor=True)
    emb_summary = model.encode(cv_summary, convert_to_tensor=True)

    raw_sim = float(util.cos_sim(emb_query, emb_summary)[0][0].cpu().numpy())
    return scale_legacy_score(raw_sim)


def get_similarity_score(
    job_query: str,
    raw_text: str,
    nlp: spacy.language.Language = None
) -> float:
    """
    Calculates document similarity using spaCy en_core_web_md between job query and raw CV text.
    
    Parameters:
        job_query (str): Job query string.
        raw_text (str): Complete raw text of the CV document.
        nlp (spacy Language, optional): Pre-initialized spaCy model instance.
        
    Returns:
        float: Document similarity score bounded between 0.0 and 1.0.
    """
    if not job_query or not raw_text:
        return 0.0

    if nlp is None:
        nlp = get_spacy_nlp()

    doc_query = nlp(job_query)
    doc_text = nlp(raw_text)

    if doc_query.vector_norm == 0 or doc_text.vector_norm == 0:
        return 0.0

    sim = float(doc_query.similarity(doc_text))
    return round(max(0.0, min(1.0, sim)), 4)
