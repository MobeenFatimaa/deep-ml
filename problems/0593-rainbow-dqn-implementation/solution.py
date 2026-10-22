import numpy as np

def rainbow_dqn_loss(
    online_V_logits: np.ndarray,
    online_A_logits: np.ndarray,
    target_V_logits: np.ndarray,
    target_A_logits: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma_n: float,
    v_min: float,
    v_max: float,
    n_atoms: int
) -> tuple:
    """
    Compute the Rainbow DQN training loss combining dueling architecture,
    double Q-learning, and distributional (categorical) Bellman projection.
    
    Returns:
        Tuple of (losses, priorities, next_actions)
    """
    batch_size = online_V_logits.shape[0]
    n_actions = online_A_logits.shape[1]
    
    # Support atoms
    z = np.linspace(v_min, v_max, n_atoms)
    delta_z = (v_max - v_min) / (n_atoms - 1) if n_atoms > 1 else 1.0
    
    # Dueling architecture
    online_mean_A = np.mean(online_A_logits, axis=1, keepdims=True)
    online_Q_logits = online_V_logits[:, None, :] + online_A_logits - online_mean_A
    
    target_mean_A = np.mean(target_A_logits, axis=1, keepdims=True)
    target_Q_logits = target_V_logits[:, None, :] + target_A_logits - target_mean_A
    
    # Softmax over atoms
    def softmax(x, axis=-1):
        ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return ex / np.sum(ex, axis=axis, keepdims=True)
        
    online_p = softmax(online_Q_logits, axis=-1)
    target_p = softmax(target_Q_logits, axis=-1)
    
    # Double Q-learning action selection
    online_Q_values = np.sum(online_p * z[None, None, :], axis=-1)
    next_actions = np.argmax(online_Q_values, axis=1).tolist()
    
    # Projection onto support
    m = np.zeros((batch_size, n_atoms))
    
    for b in range(batch_size):
        r = rewards[b]
        done = dones[b]
        a_star = next_actions[b]
        target_dist = target_p[b, a_star, :]
        
        if done:
            Tz = np.clip(r, v_min, v_max)
            beta = (Tz - v_min) / delta_z if delta_z > 0 else 0.0
            l = int(np.floor(beta))
            u = min(l + 1, n_atoms - 1)
            l = min(l, n_atoms - 1)
            if l == u:
                m[b, l] += 1.0
            else:
                m[b, l] += 1.0 * (u - beta)
                m[b, u] += 1.0 * (beta - l)
        else:
            for j in range(n_atoms):
                Tz = np.clip(r + gamma_n * z[j], v_min, v_max)
                beta = (Tz - v_min) / delta_z if delta_z > 0 else 0.0
                l = int(np.floor(beta))
                u = min(l + 1, n_atoms - 1)
                l = min(l, n_atoms - 1)
                p_mass = target_dist[j]
                if l == u:
                    m[b, l] += p_mass
                else:
                    m[b, l] += p_mass * (u - beta)
                    m[b, u] += p_mass * (beta - l)
                
    # Loss and priority computation
    losses = []
    priorities = []
    for b in range(batch_size):
        a_taken = int(actions[b])
        p_taken = online_p[b, a_taken, :]
        loss_b = -np.sum(m[b] * np.log(p_taken + 1e-8))
        priority_b = loss_b + 1e-6
        losses.append(round(float(loss_b), 4))
        priorities.append(round(float(priority_b), 4))
        
    return losses, priorities, next_actions