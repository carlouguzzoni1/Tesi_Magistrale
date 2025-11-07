import numpy as np



def dominates(s1, s2):
    """
    Checks if s1 dominates s2.

    Args:
        s1 (array): An array of solutions.
        s2 (array): An array of solutions.

    Returns:
        bool: True if s1 dominates s2, False otherwise.
    """
    
    is_better_or_equal = all(s1 <= s2)
    is_strictly_better = any(s1 < s2)
    
    return is_better_or_equal and is_strictly_better