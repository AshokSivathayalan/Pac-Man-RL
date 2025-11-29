import time, csv, math, os, pathlib, pickle
from collections import defaultdict
import numpy as np
from pacman_env.envs.pacman_world import PacManWorld
from qpolicy import encode_state
from pacman_env.envs.pacman_world import Cells

# Config (same style as Q-learning)
EPISODES = 20000
MAX_STEPS = 500
EVAL_EVERY = 100
EVAL_EPS = 10
ALPHA = 0.01
GAMMA = 0.99
EPS_START = 0.20
EPS_END = 0.005
EPS_DECAY_TAU = 20000
LOG_DIR = "runs/sarsa"

LOGDIR = "runs/sarsa"
GAMMAS = [0.8, 0.9, 0.95, 0.99]
ALPHAS = [0.15, 0.1, 0.05, 0.01]
EPSILON_STARTS=[0.4, 0.25, 0.1, 0.05, 0.01]
MAX_EPISODES=[1000, 10000, 50000]


# --- epsilon decay schedule ---
def epsilon_for_ep(ep, eps_start=EPS_START, eps_end=EPS_END, eps_decay_tau=EPS_DECAY_TAU):
    return max(eps_end, eps_start * math.exp(-ep / eps_decay_tau))


# --- ε-greedy policy ---
def epsilon_greedy(Q, n_actions, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, n_actions)
    else:
        best = np.flatnonzero(Q[state] == Q[state].max())
        return int(rng.choice(best))


# --- evaluation ---
def evaluate(env, Q, episodes=EVAL_EPS, max_steps=MAX_STEPS, seed_base=999):
    returns, pellets_eaten, survival_steps, collisions = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        done = False
        ep_ret, steps, hit = 0.0, 0, 0

        while not done and steps < max_steps:
            s = encode_state(obs)
            qvals = Q.get(s)
            if qvals is None:
                a = env.action_space.sample()
            else:
                best = np.flatnonzero(qvals == qvals.max())
                a = int(np.random.default_rng(seed_base).choice(best))

            obs, r, terminated, truncated, info = env.step(a)
            ep_ret += r
            steps += 1
            if terminated and r <= -5:
                hit += 1
            done = terminated or truncated

        returns.append(ep_ret)
        pellets_eaten.append(env.eaten)
        survival_steps.append(steps)
        collisions.append(hit)

    return {
        "avg_return": float(np.mean(returns)),
        "avg_pellets": float(np.mean(pellets_eaten)),
        "avg_steps": float(np.mean(survival_steps)),
        "avg_collisions": float(np.mean(collisions)),
    }


def train_sarsa(
    env, 
    episodes=EPISODES, 
    alpha=ALPHA, 
    gamma=GAMMA, 
    eps_start=EPS_START, 
    eps_end=EPS_END, 
    eps_decay_tau=EPS_DECAY_TAU, 
    max_steps=MAX_STEPS, 
    eval_every=EVAL_EVERY,
    eval_eps=EVAL_EPS,
    run_seed=0, 
    log_dir=LOG_DIR,
    save_path=None
):
    # Setup logging and saving directories
    os.makedirs(log_dir, exist_ok=True)
    if save_path:
        log_filename = save_path.replace(".pkl", ".csv")
    else:
        log_filename = "train_default.csv"
    logfile = os.path.join(log_dir, log_filename)

    rng = np.random.default_rng(run_seed)
    n_actions = env.action_space.n
    # Initialize Q-table with a positive small value for optimistic exploration
    Q = defaultdict(lambda: np.full(n_actions, 0.5, dtype=np.float32)) 

    with open(logfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            #"episode", "epsilon", "return", "steps", "pellets_eaten", "collisions",
            "eval_avg_return", "eval_avg_pellets", "eval_avg_steps", "eval_avg_collisions"
        ])

        for ep in range(episodes):
            # Pass decay parameters to epsilon calculation
            eps = epsilon_for_ep(ep, eps_start, eps_end, eps_decay_tau)
            
            obs, info = env.reset(seed=run_seed + ep)
            s = encode_state(obs)
            a = epsilon_greedy(Q, n_actions, s, eps, rng)

            done = False
            ep_ret, steps, ate, hit = 0.0, 0, 0, 0

            while not done and steps < max_steps:
                obs2, r, terminated, truncated, info = env.step(a)
                s2 = encode_state(obs2)
                a2 = epsilon_greedy(Q, n_actions, s2, eps, rng)

                # SARSA update: Q(s, a) = Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
                if not terminated:
                    td_target = r + gamma * Q[s2][a2]
                else:
                    td_target = r
                
                Q[s][a] += alpha * (td_target - Q[s][a])

                ep_ret += r
                if r > 0:
                    ate += 1
                if terminated and r <= -5:
                    hit += 1

                s, a = s2, a2
                done = terminated or truncated
                steps += 1

            # periodic evaluation
            eval_stats = {"avg_return": "", "avg_pellets": "", "avg_steps": "", "avg_collisions": ""}
            if (ep + 1) % eval_every == 0:
                eval_stats = evaluate(env, Q, episodes=eval_eps, max_steps=max_steps)

                w.writerow([
                    #ep + 1, f"{eps:.4f}", f"{ep_ret:.3f}", steps, ate, hit,
                    eval_stats["avg_return"], eval_stats["avg_pellets"],
                    eval_stats["avg_steps"], eval_stats["avg_collisions"]
                ])
                f.flush()

    # Save Q-table
    """ if save_path:
        pathlib.Path("models").mkdir(parents=True, exist_ok=True)
        with open(os.path.join("models", save_path), "wb") as f:
            pickle.dump(dict(Q), f)
             """
    print(f"Finished training for {save_path if save_path else 'default run'}. Final Q-table has {len(Q)} states.")
    return Q, logfile


def try_hyperparams(gammas, alphas, eps_starts):
    
    env = PacManWorld(render_mode=None) 
    algorithm = "sarsa_"
    
    for gamma in gammas:
        name = f"{algorithm}gamma_{gamma:.2f}.pkl"
        print(f"Gamma = {gamma}")
        # Train with a fixed, lower number of episodes for quick comparisons
        train_sarsa(env, episodes=50000, gamma=gamma, save_path=name) 

    for alpha in alphas:
        name = f"{algorithm}alpha_{alpha:.2f}.pkl"
        print(f"Alpha = {alpha}")
        train_sarsa(env, episodes=50000, alpha=alpha, save_path=name)

    for eps_start in eps_starts:
        name = f"{algorithm}epsstart_{eps_start:.2f}.pkl"
        print(f"Epsilon Start = {eps_start}")
        train_sarsa(env, episodes=50000, eps_start=eps_start, eps_end=0.0, save_path=name)

    env.close()

if __name__ == "__main__":
    t0 = time.time()
    #env = PacManWorld(render_mode=None) 
    #train_sarsa(env, save_path="test.pkl")
    #Q, path = train_sarsa()
    #print(f"done in {time.time() - t0:.1f}s")

    try_hyperparams(GAMMAS, ALPHAS, EPSILON_STARTS)
    
    print(f"\nAll runs done in {time.time()-t0:.1f}s")