from typing import List, Dict, Union
import numpy as np
from scipy.stats import kendalltau, spearmanr

def calculate_kendall_tau(
    ai_ranks: List[Union[int, float]],
    recruiter_ranks: List[Union[int, float]]
) -> float:
    """
    Calculates Kendall's Tau (tau) rank correlation coefficient between AI ranks and recruiter ground-truth ranks.
    
    Parameters:
        ai_ranks (list): Predicted ordinal candidate ranks (1 = best).
        recruiter_ranks (list): Recruiter ground-truth ordinal candidate ranks.
        
    Returns:
        float: Kendall's Tau statistic (-1.0 to 1.0). Returns 0.0 for invalid/empty inputs.
    """
    if len(ai_ranks) < 2 or len(recruiter_ranks) < 2 or len(ai_ranks) != len(recruiter_ranks):
        return 0.0
    try:
        tau, _ = kendalltau(ai_ranks, recruiter_ranks)
        return round(float(tau) if not np.isnan(tau) else 0.0, 4)
    except Exception as e:
        print(f"Kendall Tau calculation notice: {e}")
        return 0.0


def calculate_spearman_rho(
    ai_ranks: List[Union[int, float]],
    recruiter_ranks: List[Union[int, float]]
) -> float:
    """
    Calculates Spearman rank correlation coefficient (rho) between AI ranks and recruiter ground-truth ranks.
    
    Parameters:
        ai_ranks (list): Predicted ordinal candidate ranks (1 = best).
        recruiter_ranks (list): Recruiter ground-truth ordinal candidate ranks.
        
    Returns:
        float: Spearman Rho statistic (-1.0 to 1.0). Returns 0.0 for invalid/empty inputs.
    """
    if len(ai_ranks) < 2 or len(recruiter_ranks) < 2 or len(ai_ranks) != len(recruiter_ranks):
        return 0.0
    try:
        rho, _ = spearmanr(ai_ranks, recruiter_ranks)
        return round(float(rho) if not np.isnan(rho) else 0.0, 4)
    except Exception as e:
        print(f"Spearman Rho calculation notice: {e}")
        return 0.0


def evaluate_pipeline_rankings(
    ai_scores: List[float],
    recruiter_ranks: List[int]
) -> Dict[str, float]:
    """
    Converts predicted continuous AI scores to ordinal ranks (1 = highest score) and evaluates
    Kendall's Tau and Spearman Rho correlations against recruiter ground-truth.
    
    Parameters:
        ai_scores (list of float): Predicted AI overall scores for candidates.
        recruiter_ranks (list of int): Recruiter ground-truth ordinal ranks.
        
    Returns:
        Dict[str, float]: {"kendall_tau": float, "spearman_rho": float}
    """
    if not ai_scores or not recruiter_ranks or len(ai_scores) != len(recruiter_ranks):
        return {"kendall_tau": 0.0, "spearman_rho": 0.0}

    scores_arr = np.array(ai_scores)
    # Higher score -> rank 1 (rank 1 is best)
    # argsort twice on negated scores gives 1-based ordinal rank
    ai_ranks = (np.argsort(np.argsort(-scores_arr)) + 1).tolist()

    tau = calculate_kendall_tau(ai_ranks, recruiter_ranks)
    rho = calculate_spearman_rho(ai_ranks, recruiter_ranks)

    return {
        "kendall_tau": tau,
        "spearman_rho": rho
    }
