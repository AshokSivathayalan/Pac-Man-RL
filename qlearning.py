import time, csv, math, os
from collections import defaultdict
import numpy as np
from pacman_env.envs.pacman_world import PacManWorld, Cells
import pickle, pathlib
from qpolicy import encode_state

# imports env
# inits a Q-table - dict mapping every state to a list of action vals
# runs thousands of episodes: takes an action using ε-greedy exploration and updates Q table using Q learning rule
# short eval every 100 episodes to see how well policy performs
# logs everything into csv file

# config (tweaks here)
EPISODES = 3000
MAX_STEPS = 500
EVAL_EVERY = 100
EVAL_EPS = 10
ALPHA = 0.10    # learning rate
GAMMA = 0.95    # discount
EPS_START = 0.20
EPS_END = 0.08
EPS_DECAY_TAU = 2000
LOG_DIR = "runs/qlearning"

# control show often pacman makes random moves
def epsilon_for_ep(ep):
    # exponential decay to EPS_END
    return max(EPS_END, EPS_START * math.exp(-ep/EPS_DECAY_TAU))

# every 100 episodes run the current learned policy with no exploration to see how well it performs
def evaluate(env, Q, episodes=EVAL_EPS, max_steps=MAX_STEPS, seed_base=999):
    returns, pellets_eaten, survival_steps, collisions = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base+ep)
        done = False
        ep_ret, steps, ate, hit = 0.0, 0, 0, 0
        
        while not done and steps < max_steps:
            s = encode_state(obs)
            
            # greedy action
            qvals = Q.get(s)
            if qvals is None:
                a = env.action_space.sample()
            else:
                #a = int(np.argmax(qvals))
                best = np.flatnonzero(qvals == qvals.max())
                a = int(np.random.default_rng(seed_base).choice(best)) # random tie break among max actions
                
            obs, r, terminated, truncated, info = env.step(a)
            ep_ret += r
            steps += 1
            
            # count pellets eaten if pos rewards +1 used
            if r > 0:
                ate += int(r)
            # count collisions if -5 on death
            if terminated and r <= -5:
                hit += 1
                
            done = terminated or truncated
            
        returns.append(ep_ret)
        pellets_eaten.append(ate)
        survival_steps.append(steps)
        collisions.append(hit)
        
    out = {
        "avg_return": float(np.mean(returns)),
        "avg_pellets": float(np.mean(pellets_eaten)),
        "avg_steps": float(np.mean(survival_steps)),
        "avg_collisions": float(np.mean(collisions))
    }
    return out

def train_once(run_seed=0, log_dir=LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"train.csv")
    
    env = PacManWorld(render_mode=None)
    rng = np.random.default_rng(run_seed)
    
    # Q table: dict[state_key] -> np.array(n_actions)
    n_actions = env.action_space.n
    # optimistic init to discourage stalling
    Q = defaultdict(lambda: np.full(n_actions, 0.5, dtype=np.float32))
    
    # csv logger
    with open(logfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode","epsilon","return","steps","pellets_eaten","collisions",
            "eval_avg_return","eval_avg_pellets","eval_avg_steps","eval_avg_collisions"
        ])
        
        for ep in range(EPISODES):
            eps = epsilon_for_ep(ep)
            obs, info = env.reset(seed=run_seed + ep)
            done = False

            ep_ret, steps, ate, hit = 0.0, 0, 0, 0

            while not done and steps < MAX_STEPS:
                s = encode_state(obs)

                # ε-greedy action with tie break
                if np.random.rand() < eps:
                    a = env.action_space.sample()
                else:
                    best = np.flatnonzero(Q[s] == Q[s].max())
                    a = int(rng.choice(best))
                
                    #a = int(np.argmax(Q[s]))

                obs2, r, terminated, truncated, info = env.step(a)
                s2 = encode_state(obs2)

                # Q-learning update
                Q[s][a] += ALPHA * (r + GAMMA * np.max(Q[s2]) - Q[s][a])

                ep_ret += r
                steps += 1
                if r > 0:
                    ate += 1
                if terminated and r <= -5:
                    hit += 1

                obs = obs2
                done = terminated or truncated

            # periodic evaluation (ε≈0 policy)
            eval_stats = {"avg_return": "", "avg_pellets": "", "avg_steps": "", "avg_collisions": ""}
            if (ep + 1) % EVAL_EVERY == 0:
                eval_stats = evaluate(env, Q, episodes=EVAL_EPS)

            w.writerow([
                ep + 1, f"{eps:.4f}", f"{ep_ret:.3f}", steps, ate, hit,
                eval_stats["avg_return"], eval_stats["avg_pellets"],
                eval_stats["avg_steps"], eval_stats["avg_collisions"]
            ])
            
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    with open(os.path.join("models", f"qlearning.pkl"),"wb") as f:
        pickle.dump(dict(Q), f)

    env.close()
    return Q, logfile

if __name__ == "__main__":
    t0 = time.time()
    Q, path = train_once(run_seed=0, log_dir=LOG_DIR)
    print(f"done in {time.time()-t0:.1f}s")
    