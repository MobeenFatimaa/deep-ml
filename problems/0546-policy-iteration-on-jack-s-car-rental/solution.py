import numpy as np
from math import factorial, exp


def compute_poisson_probs(lam, upper_bound):
    probs = np.zeros(upper_bound)
    for k in range(upper_bound):
        probs[k] = exp(-lam) * (lam**k) / factorial(k)
    return probs


def get_feasible_actions(i, j, max_cars, max_move):
    low = max(-max_move, i - max_cars, -j)
    high = min(max_move, i, max_cars - j)
    return list(range(low, high + 1))


def compute_expected_return(
    i,
    j,
    a,
    value,
    max_cars,
    req_probs,
    ret_probs,
    rental_credit,
    move_cost,
    gamma,
):
    i_after = i - a
    j_after = j + a
    move_reward = -move_cost * abs(a)

    req1_probs, req2_probs = req_probs
    ret1_probs, ret2_probs = ret_probs

    upper = len(req1_probs)
    req1 = np.arange(upper)
    req2 = np.arange(upper)
    ret1 = np.arange(upper)
    ret2 = np.arange(upper)

    rent1 = np.minimum(i_after, req1[:, None, None, None])
    rent2 = np.minimum(j_after, req2[None, :, None, None])
    revenue = (rent1 + rent2) * rental_credit

    i_next = np.clip(i_after - rent1 + ret1[None, None, :, None], 0, max_cars)
    j_next = np.clip(j_after - rent2 + ret2[None, None, None, :], 0, max_cars)

    prob_tensor = (
        req1_probs[:, None, None, None]
        * req2_probs[None, :, None, None]
        * ret1_probs[None, None, :, None]
        * ret2_probs[None, None, None, :]
    )

    V_arr = np.array(value)
    V_next = V_arr[i_next, j_next]

    expected_return = move_reward + np.sum(
        prob_tensor * (revenue + gamma * V_next)
    )
    return float(expected_return)


def jacks_car_rental(
    max_cars,
    max_move,
    rental_credit,
    move_cost,
    request_lambda,
    return_lambda,
    gamma,
    poisson_upper_bound=11,
    theta=0.01,
):
    """Solve a simplified car rental MDP using policy iteration.

    Args:
        max_cars: int, maximum cars at each location
        max_move: int, maximum cars moved per night
        rental_credit: float, revenue per rental
        move_cost: float, cost per car moved
        request_lambda: tuple (lambda1, lambda2) for Poisson request rates
        return_lambda: tuple (lambda1, lambda2) for Poisson return rates
        gamma: float, discount factor
        poisson_upper_bound: int, Poisson truncation (default 11)
        theta: float, policy evaluation convergence threshold (default 0.01)

    Returns:
        tuple: (policy, value, num_iterations)
            policy: list of lists of int, optimal cars to move from loc1 to loc2
            value: list of lists of float, rounded to 2 decimals
            num_iterations: int, number of policy improvement steps
    """
    req_probs = (
        compute_poisson_probs(request_lambda[0], poisson_upper_bound),
        compute_poisson_probs(request_lambda[1], poisson_upper_bound),
    )
    ret_probs = (
        compute_poisson_probs(return_lambda[0], poisson_upper_bound),
        compute_poisson_probs(return_lambda[1], poisson_upper_bound),
    )

    n_states = max_cars + 1
    policy = [[0 for _ in range(n_states)] for _ in range(n_states)]
    value = [[0.0 for _ in range(n_states)] for _ in range(n_states)]

    num_iterations = 0

    while True:
        # Policy Evaluation
        while True:
            max_delta = 0.0
            for i in range(n_states):
                for j in range(n_states):
                    old_v = value[i][j]
                    a = policy[i][j]
                    new_v = compute_expected_return(
                        i,
                        j,
                        a,
                        value,
                        max_cars,
                        req_probs,
                        ret_probs,
                        rental_credit,
                        move_cost,
                        gamma,
                    )
                    value[i][j] = new_v
                    max_delta = max(max_delta, abs(old_v - new_v))
            if max_delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for i in range(n_states):
            for j in range(n_states):
                old_action = policy[i][j]
                feasible = get_feasible_actions(i, j, max_cars, max_move)

                best_action = feasible[0]
                best_val = -float("inf")

                for a in feasible:
                    val = compute_expected_return(
                        i,
                        j,
                        a,
                        value,
                        max_cars,
                        req_probs,
                        ret_probs,
                        rental_credit,
                        move_cost,
                        gamma,
                    )
                    if val > best_val:
                        best_val = val
                        best_action = a
                    elif np.isclose(val, best_val):
                        if a < best_action:
                            best_action = a

                policy[i][j] = best_action
                if best_action != old_action:
                    policy_stable = False

        num_iterations += 1
        if policy_stable:
            break

    rounded_value = [[round(val, 2) for val in row] for row in value]
    return policy, rounded_value, num_iterations