import numpy as np

class GradientBandit:
    def __init__(self, num_actions, alpha=0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = np.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0
        
    def softmax(self):
        # Compute softmax probabilities from preferences with numerical stability
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))
        return exp_prefs / np.sum(exp_prefs)
        
    def select_action(self):
        # Sample an action according to the softmax distribution
        probs = self.softmax()
        return int(np.random.choice(self.num_actions, p=probs))
        
    def update(self, action, reward):
        # Update action preferences using the gradient ascent update
        self.time += 1
        
        # Update average reward baseline incrementally
        self.avg_reward += (reward - self.avg_reward) / self.time
        
        # Get current action probabilities
        probs = self.softmax()
        
        # Create one-hot vector for the selected action
        one_hot = np.zeros(self.num_actions)
        one_hot[action] = 1.0
        
        # Update preferences via gradient ascent:
        # H(a) <- H(a) + alpha * (R - R_bar) * (I(a == A_t) - pi(a))
        self.preferences += self.alpha * (reward - self.avg_reward) * (one_hot - probs)