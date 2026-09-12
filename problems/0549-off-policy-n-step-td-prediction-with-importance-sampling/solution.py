import numpy as np

def off_policy_nstep_td(episodes, behavior_policy, target_policy, num_states, num_actions, n, alpha, gamma):
    V = np.zeros(num_states, dtype=float)

    for episode in episodes:
        T = len(episode)
        
        for t in range(T):
            state_t, _, _ = episode[t]
            
            # 1. Compute n-step return G
            G = 0.0
            max_k = min(n, T - t)
            
            for k in range(max_k):
                _, _, reward = episode[t + k]
                G += (gamma ** k) * reward
                
            if t + n < T:
                next_state = episode[t + n][0]
                G += (gamma ** n) * V[next_state]
                
            # 2. Importance sampling ratio over the n-step window
            rho = 1.0
            for i in range(t, min(t + n, T)):
                s_i, a_i, _ = episode[i]
                pi_prob = target_policy[s_i][a_i]
                b_prob = behavior_policy[s_i][a_i]
                
                rho *= (pi_prob / b_prob) if b_prob > 0 else 0.0
                
            # 3. Apply TD update
            V[state_t] += alpha * rho * (G - V[state_t])
            
    return V