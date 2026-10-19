import numpy as np

def acrobot_simulate(initial_state: list, actions: list, dt: float = 0.2) -> tuple:
    """
    Simulate the Acrobot swing-up environment.
    
    Args:
        initial_state: [theta1, theta2, dtheta1, dtheta2]
        actions: list of torque actions (-1, 0, or 1)
        dt: integration time step
    
    Returns:
        (states, total_reward, done_step)
    """
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    lc1 = 0.5
    lc2 = 0.5
    I1 = 1.0
    I2 = 1.0
    g = 9.8
    
    max_dtheta1 = 4 * np.pi
    max_dtheta2 = 9 * np.pi
    
    torque_map = {-1: -1.0, 0: 0.0, 1: 1.0}
    
    state = np.array(initial_state, dtype=float)
    
    def round_state(s):
        rounded = [round(float(x), 4) for x in s]
        return [0.0 if x == -0.0 else x for x in rounded]
        
    states = [round_state(state)]
    total_reward = 0.0
    done_step = -1
    
    for step_idx, a in enumerate(actions, 1):
        theta1, theta2, dtheta1, dtheta2 = state
        tau = torque_map[a]
        
        d1 = m1 * (lc1 ** 2) + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2)
        phi1 = (-m2 * l1 * lc2 * (dtheta2 ** 2) * np.sin(theta2) 
                - 2 * m2 * l1 * lc2 * dtheta1 * dtheta2 * np.sin(theta2) 
                + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) 
                + phi2)
                
        denominator = m2 * (lc2 ** 2) + I2 - (d2 ** 2) / d1
        ddtheta2 = (tau + (d2 / d1) * phi1 - m2 * l1 * lc2 * (dtheta1 ** 2) * np.sin(theta2) - phi2) / denominator
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        
        new_dtheta1 = dtheta1 + ddtheta1 * dt
        new_dtheta2 = dtheta2 + ddtheta2 * dt
        
        new_dtheta1 = np.clip(new_dtheta1, -max_dtheta1, max_dtheta1)
        new_dtheta2 = np.clip(new_dtheta2, -max_dtheta2, max_dtheta2)
        
        new_theta1 = theta1 + 0.5 * (dtheta1 + new_dtheta1) * dt
        new_theta2 = theta2 + 0.5 * (dtheta2 + new_dtheta2) * dt
        
        new_theta1 = (new_theta1 + np.pi) % (2 * np.pi) - np.pi
        new_theta2 = (new_theta2 + np.pi) % (2 * np.pi) - np.pi
        
        state = np.array([new_theta1, new_theta2, new_dtheta1, new_dtheta2])
        states.append(round_state(state))
        total_reward -= 1.0
        
        tip_height = -np.cos(new_theta1) - np.cos(new_theta1 + new_theta2)
        if tip_height > 1.0 and done_step == -1:
            done_step = step_idx
            break
            
    return states, total_reward, done_step