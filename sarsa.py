import gymnasium
import numpy as np
import pickle
from collections import defaultdict
from qpolicy import encode_state
import time, pathlib, os

LOGDIR="runs/sarsa"

def sarsa(env, gamma=0.95, alpha=0.1, epsilon=0.1, max_episode=30000):
    S = 0
    A = env.action_space.n
    Q = defaultdict(lambda: np.full(A, 0.5, dtype=np.float32))
    # Terminal state should have value 0?
    
    for i in range(max_episode):
        # Begin episode
        obs, _ = env.reset()
        state = encode_state(obs)
        # Pick action according to epsilon-greedy policy
        action = epsilon_greedy(Q, A, state, epsilon)
        terminated = truncated = False

        # Loop through episode
        while not terminated and not truncated:
            # Act, get reward
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(obs)
            # determine next action
            next_act = epsilon_greedy(Q, A, next_state, epsilon)

            # update action values
            if not terminated:
                Q[state][action] += alpha*(reward + gamma*Q[next_state][next_act] -Q[state][action])
            else:
                Q[state][action] += alpha*(reward - Q[state][action])
            
            state = next_state
            action = next_act

    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    with open(os.path.join("models", f"sarsa.pkl"),"wb") as f:
        pickle.dump(dict(Q), f)

    env.close()
    return Q

def epsilon_greedy(Q, A, state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.randint(0, A)
    else:
        return np.argmax(Q[state])
    

if __name__ == "__main__":
    t0 = time.time()
    env = gymnasium.make('pacman_env/PacManWorld')
    Q = sarsa(env)
    print(f"done in {time.time()-t0:.1f}s")