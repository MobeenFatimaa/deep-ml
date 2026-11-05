import numpy as np


def continuing_actor_critic(next_state, reward, s0, uniforms, alpha_w, alpha_theta, alpha_rbar, lam_w, lam_theta):
    """
    Implements tabular Average-Reward Actor-Critic with eligibility traces for a continuing task.
    
    Args:
        next_state: array of shape (n_states, n_actions) giving next states
        reward: array of shape (n_states, n_actions) giving rewards
        s0: initial state (int)
        uniforms: list or array of uniform random numbers in [0, 1) for action selection
        alpha_w: step size for critic
        alpha_theta: step size for actor
        alpha_rbar: step size for average reward
        lam_w: trace decay parameter for critic
        lam_theta: trace decay parameter for actor
        
    Returns:
        (theta, w, rbar, r_pi): final preferences, critic weights, average reward estimate, and exact steady-state average reward
    """
    n_states, n_actions = next_state.shape
    
    theta = np.zeros((n_states, n_actions))
    w = np.zeros(n_states)
    rbar = 0.0
    
    zw = np.zeros(n_states)
    ztheta = np.zeros((n_states, n_actions))
    
    current_state = s0
    
    def get_policy(th):
        exp_th = np.exp(th - np.max(th, axis=1, keepdims=True))
        return exp_th / np.sum(exp_th, axis=1, keepdims=True)
    
    for u in uniforms:
        pi = get_policy(theta)
        
        # Choose action by inverse-CDF sampling
        cum_prob = np.cumsum(pi[current_state])
        action = 0
        for a in range(n_actions):
            if cum_prob[a] > u:
                action = a
                break
        if action >= n_actions:
            action = n_actions - 1
            
        next_s = next_state[current_state, action]
        r = reward[current_state, action]
        
        # TD error for average reward
        delta = r - rbar + w[next_s] - w[current_state]
        
        # Update average reward estimate
        rbar += alpha_rbar * delta
        
        # Gradients
        grad_w = np.zeros(n_states)
        grad_w[current_state] = 1.0
        
        pi_s = pi[current_state]
        grad_theta = -pi_s
        grad_theta[action] += 1.0
        
        # Update eligibility traces
        zw = lam_w * zw + grad_w
        ztheta = lam_theta * ztheta
        ztheta[current_state] += grad_theta
        
        # Update critic and actor weights
        w += alpha_w * delta * zw
        theta += alpha_theta * delta * ztheta
        
        current_state = next_s
        
    # Compute exact steady-state average reward r(pi) for final policy
    pi_final = get_policy(theta)
    P = np.zeros((n_states, n_states))
    expected_rewards = np.zeros(n_states)
    
    for s in range(n_states):
        for a in range(n_actions):
            ns = next_state[s, a]
            r_val = reward[s, a]
            prob = pi_final[s, a]
            P[s, ns] += prob
            expected_rewards[s] += prob * r_val
            
    # Solve linear system for stationary distribution mu: (P.T - I) * mu = 0 s.t. sum(mu) = 1
    A = np.vstack([(P.T - np.eye(n_states)), np.ones(n_states)])
    b = np.append(np.zeros(n_states), 1.0)
    mu = np.linalg.lstsq(A, b, rcond=None)[0]
    
    r_pi = float(np.sum(mu * expected_rewards))
    
    return theta, w, rbar, r_pi