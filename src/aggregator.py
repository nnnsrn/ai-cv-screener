from typing import Dict, Optional

# Default weights for Proposed Hybrid Pipeline (Semantic + Similarity + Structurized)
DEFAULT_HYBRID_WEIGHTS = {
    "semantic": 0.3,
    "similarity": 0.2,
    "structurized": 0.5
}

# Default weights for Legacy Pipeline (Semantic + Similarity)
DEFAULT_LEGACY_WEIGHTS = {
    "semantic": 0.6,
    "similarity": 0.4,
    "structurized": 0.0
}


def calculate_final_score(
    semantic_score: float,
    similarity_score: float,
    structurized_score: float = 0.0,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculates the final score as a configurable weighted sum across semantic, similarity, and structurized scores.

    Parameters:
        semantic_score (float): BAAI/bge-m3 cosine similarity score (scaled).
        similarity_score (float): spaCy document text similarity score.
        structurized_score (float): Aggregated quantitative + qualitative score (0.0 for legacy).
        weights (dict, optional): Custom dictionary containing 'semantic', 'similarity', 'structurized' weights.

    Returns:
        float: Weighted final score bounded between 0.0 and 1.0.
    """
    if weights is None:
        if structurized_score > 0.0:
            weights = DEFAULT_HYBRID_WEIGHTS
        else:
            weights = DEFAULT_LEGACY_WEIGHTS

    w_sem = max(0.0, float(weights.get("semantic", 0.3)))
    w_sim = max(0.0, float(weights.get("similarity", 0.2)))
    w_str = max(0.0, float(weights.get("structurized", 0.5)))

    # Normalize weights so sum equals 1.0
    total_w = w_sem + w_sim + w_str
    if total_w > 0:
        w_sem /= total_w
        w_sim /= total_w
        w_str /= total_w
    else:
        w_sem, w_sim, w_str = 0.3333, 0.3333, 0.3334

    final_score = (w_sem * semantic_score) + (w_sim * similarity_score) + (w_str * structurized_score)
    return round(float(max(0.0, min(1.0, final_score))), 4)
