"""
differential_privacy.py

Adds differential privacy by injecting Gaussian noise to updates.
"""

import numpy as np

def clip_gradients(update, clipping_norm):
    """
    Clip the gradient to the given norm.
    """
    norm = np.linalg.norm(update)
    if norm > clipping_norm:
        return update * (clipping_norm / norm)
    return update

def add_dp_noise(update, clipping_norm, noise_std=0.1):
    """
    Adds Gaussian noise to the update for differential privacy.

    Parameters:
      update (np.array): Aggregated update.
      clipping_norm (float): Max L2 norm for each individual update.noise_std (float): Standard deviation of the Gaussian noise.
      
    Returns:
      np.array: Noisy update.
    """
    noise = np.random.normal(0, scale=clipping_norm * noise_std, size=update.shape)
    return update + noise

def differentially_private_aggregation(updates, clipping_norm=1.0, noise_std=1.0):
    """
    Federated averaging with enhanced differential privacy.
    
    Args:
        updates (List[np.ndarray]): List of model updates from clients.
        clipping_norm (float): Max L2 norm for each individual update. Default value of 1.0.
        noise_std (float): Controls noise magnitude for DP. Default value of 1.0.

    Returns:
        np.ndarray: Differentially private averaged update.
    """
    clipped_updates = [clip_gradients(update, clipping_norm) for update in updates]
    averaged = np.mean(clipped_updates, axis=0)
    noised = add_dp_noise(averaged, clipping_norm, noise_std)
    return noised