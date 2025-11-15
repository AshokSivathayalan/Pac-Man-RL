import time, csv, math, os, pathlib, pickle
from collections import defaultdict
import numpy as np
from pacman_env.envs.pacman_world import PacManWorld
from qpolicy import encode_state

# Config (same style as Q-learning)
EPISODES = 3000
MAX_STEPS = 500
EVAL_EVERY = 100
EVAL_EPS = 10
ALPHA = 0.10
GAMMA = 0.95
EPS_START = 0.20
EPS_END = 0.08
EPS_DECAY_TAU = 2000
LOG_DIR = "runs/sarsa"


# --- epsilon decay schedule ---
def epsilon_for_ep(ep):
    return max(EPS_END, EPS_START * math.exp(-ep / EPS_DECAY_TAU))


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
        ep_ret, steps, ate, hit = 0.0, 0, 0, 0

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
            if r > 0:
                ate += int(r)
            if terminated and r <= -5:
                hit += 1
            done = terminated or truncated

        returns.append(ep_ret)
        pellets_eaten.append(ate)
        survival_steps.append(steps)
        collisions.append(hit)

    return {
        "avg_return": float(np.mean(returns)),
        "avg_pellets": float(np.mean(pellets_eaten)),
        "avg_steps": float(np.mean(survival_steps)),
        "avg_collisions": float(np.mean(collisions)),
    }


# --- SARSA training loop ---
def train_sarsa(run_seed=0, log_dir=LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, "train.csv")

    env = PacManWorld(render_mode=None)
    rng = np.random.default_rng(run_seed)

    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.full(n_actions, 0.5, dtype=np.float32))

    with open(logfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "epsilon", "return", "steps", "pellets_eaten", "collisions",
            "eval_avg_return", "eval_avg_pellets", "eval_avg_steps", "eval_avg_collisions"
        ])

        for ep in range(EPISODES):
            eps = epsilon_for_ep(ep)
            obs, info = env.reset(seed=run_seed + ep)
            s = encode_state(obs)
            a = epsilon_greedy(Q, n_actions, s, eps, rng)

            done = False
            ep_ret, steps, ate, hit = 0.0, 0, 0, 0

            while not done and steps < MAX_STEPS:
                obs2, r, terminated, truncated, info = env.step(a)
                s2 = encode_state(obs2)
                a2 = epsilon_greedy(Q, n_actions, s2, eps, rng)

                # SARSA update
                if not terminated:
                    Q[s][a] += ALPHA * (r + GAMMA * Q[s2][a2] - Q[s][a])
                else:
                    Q[s][a] += ALPHA * (r - Q[s][a])

                ep_ret += r
                if r > 0:
                    ate += int(r)
                if terminated and r <= -5:
                    hit += 1

                s, a = s2, a2
                done = terminated or truncated
                steps += 1

            # periodic evaluation
            eval_stats = {"avg_return": "", "avg_pellets": "", "avg_steps": "", "avg_collisions": ""}
            if (ep + 1) % EVAL_EVERY == 0:
                eval_stats = evaluate(env, Q, episodes=EVAL_EPS)

            w.writerow([
                ep + 1, f"{eps:.4f}", f"{ep_ret:.3f}", steps, ate, hit,
                eval_stats["avg_return"], eval_stats["avg_pellets"],
                eval_stats["avg_steps"], eval_stats["avg_collisions"]
            ])

    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    with open(os.path.join("models", "sarsa.pkl"), "wb") as f:
        pickle.dump(dict(Q), f)

    env.close()
    return Q, logfile


if __name__ == "__main__":
    t0 = time.time()
    Q, path = train_sarsa()
    print(f"done in {time.time() - t0:.1f}s")
