import numpy as np

def dyna_q(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes, n_planning):
    """
    Implement the Dyna-Q algorithm.
    
    Args:
        num_states: int, number of states
        num_actions: int, number of actions
        P: np.ndarray of shape (num_states, num_actions, num_states), transition probabilities
        R: np.ndarray of shape (num_states, num_actions), rewards
        terminal_states: list of int, terminal state indices
        alpha: float, learning rate
        gamma: float, discount factor
        epsilon: float, exploration rate
        num_episodes: int, number of episodes
        n_planning: int, number of planning steps per real step
    
    Returns:
        Q: np.ndarray of shape (num_states, num_actions), learned Q-table
    """
    Q = np.zeros((num_states, num_actions), dtype=float)
    model = {}
    terminal_set = set(terminal_states)
    non_terminal_states = [s for s in range(num_states) if s not in terminal_set]
    
    def select_action(s):
        if np.random.rand() < epsilon:
            return np.random.randint(num_actions)
        else:
            # argmax with tie-breaking by choosing the smallest action index
            return int(np.argmax(Q[s]))

    for episode in range(num_episodes):
        if not non_terminal_states:
            break
        s = np.random.choice(non_terminal_states)
        
        while s not in terminal_set:
            a = select_action(s)
            
            # Real environment interaction
            probs = P[s, a]
            s_prime = np.random.choice(num_states, p=probs)
            r = R[s, a]
            
            # 1. Direct RL (Q-learning update)
            if s_prime in terminal_set:
                target = r
            else:
                target = r + gamma * np.max(Q[s_prime])
            Q[s, a] += alpha * (target - Q[s, a])
            
            # 2. Model Learning
            model[(s, a)] = (r, s_prime)
            
            # 3. Planning
            if model:
                model_keys = list(model.keys())
                for _ in range(n_planning):
                    rand_idx = np.random.choice(len(model_keys))
                    sim_s, sim_a = model_keys[rand_idx]
                    sim_r, sim_s_prime = model[(sim_s, sim_a)]
                    
                    if sim_s_prime in terminal_set:
                        sim_target = sim_r
                    else:
                        sim_target = sim_r + gamma * np.max(Q[sim_s_prime])
                    Q[sim_s, sim_a] += alpha * (sim_target - Q[sim_s, sim_a])
            
            s = s_prime
            
    return Q