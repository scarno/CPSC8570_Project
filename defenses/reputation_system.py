"""
reputation_system.py

Implements a simple reputation system that assigns and updates trust scores for clients.
"""

import numpy as np

class ReputationSystem:
    def __init__(self, num_clients, initial_score=1.0, decay=0.01, reward=0.05, penalty=0.1):
        # Initialize all clients with a neutral trust score of 1.0
        #self.trust_scores = np.ones(num_clients) * initial_score
        self.trust_scores = {client_id: 1.0 for client_id in range(num_clients)}
        #print("Trust Scores:",self.trust_scores)
        self.decay = decay
        self.reward = reward
        self.penalty = penalty

    def update(self, client_updates, validation_score):
        """
        Update trust scores based on the validation performance.
        If the global performance is poor, decrease trust; otherwise, slightly increase.
        """
        #print("updates:",client_updates)
        for client_id, _ in client_updates:
            # Decay old score
            #self.trust_scores[client_id] *= (1 - self.decay)

            # Reward or penalize
            if validation_score < 0.5:
                #self.trust_scores[client_id] *= 0.95
                self.trust_scores[client_id] -= self.penalty
            else:
                #self.trust_scores[client_id] *= 1.01
                self.trust_scores[client_id] += self.reward

            # Clamp score between 0 and 1, so the score doesn't get bigger that 1 or smaller than 0
            self.trust_scores[client_id] = np.clip(self.trust_scores[client_id], 0, 1)
            #print(f"Client {client_id+1} updates:",self.trust_scores[client_id])
        
        # Normalize trust scores for stability
        total = sum(self.trust_scores)
        #print("Total:",total)
        for client_id, _ in client_updates:
            self.trust_scores[client_id] /= total
 
    def get_trust(self, client_id):
        return self.trust_scores[client_id]
    
    def __repr__(self):
        return str(self.trust_scores)
