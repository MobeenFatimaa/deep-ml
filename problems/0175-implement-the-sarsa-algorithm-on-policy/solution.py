def sarsa_update(transitions, initial_states, alpha, gamma, max_steps):
    """
    Perform SARSA updates on the given environment transitions.
    Args:
        transitions (dict): mapping (state, action) -> (reward, next_state)
        initial_states (list): list of starting states to simulate episodes from
        alpha (float): learning rate
        gamma (float): discount factor
        max_steps (int): maximum steps allowed per episode
    Returns:
        dict: final Q-table as a dictionary {(state, action): value}
    """
    # Initialize Q-values for all state-action pairs in transitions to 0.0
    Q = {key: 0.0 for key in transitions}
    
    # Helper function to get available actions for a given state
    def get_actions(state):
        return [act for (s, act) in transitions.keys() if s == state]

    # Helper function to choose the greedy action (breaking ties alphabetically)
    def choose_action(state):
        actions = get_actions(state)
        if not actions:
            return None
        # Maximize Q(state, a), break ties by choosing the lexicographically smallest action
        best_action = min(actions, key=lambda a: ( -Q.get((state, a), 0.0), a ))
        return best_action

    # Run one episode per entry in initial_states, in order
    for s in initial_states:
        if s == 'terminal':
            continue
            
        current_state = s
        current_action = choose_action(current_state)
        if current_action is None:
            continue
            
        steps = 0
        while steps < max_steps and current_state != 'terminal':
            # Execute step
            reward, next_state = transitions[(current_state, current_action)]
            
            # Determine next action a' from next_state using the greedy policy
            if next_state == 'terminal':
                next_action = None
            else:
                next_action = choose_action(next_state)
                
            # Compute Q(s', a') target value
            q_next = 0.0 if next_action is None else Q.get((next_state, next_action), 0.0)
            
            # SARSA update rule: Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
            q_current = Q[(current_state, current_action)]
            Q[(current_state, current_action)] = q_current + alpha * (reward + gamma * q_next - q_current)
            
            # Transition to next state and action
            current_state = next_state
            current_action = next_action
            steps += 1
            
            if current_state == 'terminal' or current_action is None:
                break
                
    return Q