"""
robust_aggregation.py

Implements a robust aggregation mechanism using the Krum algorithm.
"""

import numpy as np

def krum_aggregation(updates, f=1):
    """
    Krum aggregation selects one update from the list that is most "central".

    Parameters:
      updates (list of np.array): List of client updates.
      f (int): Number of Byzantine (malicious) updates tolerated. If the function is called without an argument, then the default value of 1 will be used.
      
    Returns:
      np.array: Aggregated update.
    """
    num_updates = len(updates)
    scores = []
    
    # Compute pairwise distances between updates
    for i in range(num_updates):
        distances = []
        for j in range(num_updates):
            if i != j:
                distances.append(np.linalg.norm(updates[i] - updates[j])**2)
        # Sort distances and sum the closest (n - f - 2)
        distances.sort()
        score = sum(distances[:num_updates - f - 2])
        scores.append(score)
    
    # Select the update with the minimal score
    krum_index = int(np.argmin(scores))
    print(f"Krum selected update from client index {krum_index} with score {scores[krum_index]:.4f}")
    return updates[krum_index]
