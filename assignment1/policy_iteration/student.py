import numpy as np

import random
import numpy as np
import gymnasium as gym


def reward_probabilities(env_size):
    rewards = np.zeros((env_size*env_size))
    i = 0
    for r in range(env_size):
        for c in range(env_size):
            state = np.array([r,c], dtype=np.uint8)
            rewards[i] = reward_function(state, env_size)
            i+=1
    return rewards

def reward_function(s, env_size):
    r = 0.0
    if (s == np.array([env_size-1, env_size-1])).all():
        r = 1
    return r

def transition_probabilities(env, s, a, env_size, directions, holes):
    cells = []
    probs = []
    prob_next_state = np.zeros((env_size, env_size))

    def check_feasibility(s_prime, s, env_size):
        if (s_prime < 0).any(): return s
                
        if s_prime[0] >= env_size: return s
        if s_prime[1] >= env_size: return s

        return s_prime

    s_prime = check_feasibility(s + directions[a, :], s, env_size)
    prob_next_state[s_prime[0], s_prime[1]] += 1/2

    s_prime = check_feasibility(s + directions[(a-1) % 4, :], s, env_size)
    prob_next_state[s_prime[0], s_prime[1]] += 1/2

    return prob_next_state

def value_iteration(env, env_size, end_state, directions, obstacles, gamma=0.99, max_iters=1000, theta=1e-3):
    # initialize
    values = np.zeros(env.observation_space.n, dtype=float)
    policy = np.zeros(env.observation_space.n, dtype=int)  # argmax actions we’ll derive
    STATES = np.zeros((env.observation_space.n, 2), dtype=np.uint8)
    REWARDS = reward_probabilities(env_size)

    # enumerate grid states in the same way
    k = 0
    for r in range(env_size):
        for c in range(env_size):
            STATES[k] = np.array([r, c], dtype=np.uint8)
            k += 1


    for i in range(max_iters):
        delta = 0.0
        v_old = values.copy()

        for s in range(env.observation_space.n):
            state = STATES[s]

            # terminal (goal) or obstacle cells have no outgoing value
            done = (state == end_state).all() or obstacles[state[0], state[1]]
            if done:
                new_v = 0.0
                best_a = policy[s] 
            else:
                best_value = -float('inf')
                best_action = 0
                for a in range(env.action_space.n):
                    next_state_prob = transition_probabilities(
                        env, state, a, env_size, directions, obstacles
                    ).flatten()
                    va = (next_state_prob * (REWARDS + gamma * v_old)).sum()
                    if va > best_value:
                        best_value = va
                        best_action = a
                new_v = best_value
                best_a = best_action

            values[s] = new_v
            policy[s] = best_a
            delta = max(delta, abs(v_old[s] - values[s]))

        if delta < theta:
            break

    print(f'finished in {i+1} iterations')
    return policy.reshape((env_size, env_size)), values.reshape((env_size, env_size))

def policy_iteration(env, env_size, end_state, directions, obstacles, gamma=0.99, max_iters=1000, theta=1e-3):
    # rename to policy
    
    # policy evaluation 
    policy = np.random.randint(0, env.action_space.n, (env.observation_space.n))
    values = np.random.random(env_size*env_size)
    
    # list, where each element is a list of len 2
    STATES = np.zeros((env_size*env_size, 2), dtype=np.uint8)
    # 1-D list of size env_size*env_size, each element has dimention
    REWARDS = reward_probabilities(env_size)
    
    # number of total states
    i = 0 
    # we iterate over all the states
    for r in range(env_size):
        for c in range(env_size):
            # we save the indices like this to have faster access to the singular index
            state = np.array([r,c], dtype=np.uint8)
            STATES[i] = state
            i+=1

    # policy evaluation
    for i in range(max_iters):
        while True:
            delta = 0
            
            for s in range(len(STATES)):
                state = STATES[s]
                v_old = values[s]

                action = policy[s]
                # done is equal to 1 if both elements in state are equal to end_state element, thanks to .all()
                done = (state == end_state).all()

                next_state_prob = transition_probabilities(env, state, action, env_size, directions, obstacles).flatten()
                
                values[s] = (1-done) * (next_state_prob * (REWARDS + gamma * values)).sum()

                delta = max(delta, np.abs(v_old - values[s]))
            
            # break the while 
            if delta < theta:
                break

        # policy improvement
        policy_stable = True
        #print(f'observed: {env.observation_space.n} | states: {len(STATES)}')
        for s in range(env.observation_space.n):
            b = policy[s]
            state = STATES[s]

            best_value = -float('inf')
            best_action = None
            
            for a in range(env.action_space.n):
                next_state_prob = transition_probabilities(env, state, a, env_size, directions, obstacles).flatten()
                va = (next_state_prob * (REWARDS + gamma * values)).sum()

                if va > best_value:
                    best_value = va
                    best_action = a
            
            policy[s] = best_action

            if best_action != b:
                policy_stable = False
        
        if policy_stable:

            break
    
    return policy.reshape((env_size, env_size)), values.reshape((env_size, env_size))