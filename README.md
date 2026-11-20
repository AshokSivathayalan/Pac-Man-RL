# Pac-Man Reinforcement Learning

A custom Pac-Man environment built on [Gymnasium](https://gymnasium.farama.org/), used to train and compare two tabular reinforcement learning algorithms, **Q-Learning** and **SARSA**. The game is made harder than usual: the ghost moves **twice as fast** as Pac-Man, and eaten pellets **regenerate at random** during an episode.

Course project for COMP 4010 (Reinforcement Learning) at Carleton University, Fall 2025.

## Results

| | Q-Learning | SARSA |
|---|---|---|
| Peak pellets eaten per episode | ~75 | ~60–65 |
| Peak return | ~60+ | ~53 |
| Episodes to a stable, strong policy | ~3,000 | ~20,000 |

Q-Learning came out ahead and converged much faster. We put this down to it being off-policy: it learns the value of the best path even while exploring at random. SARSA is on-policy, so it undervalues risky paths, and the winning strategy was a risky one.

Both agents learned to **exploit the ghost's chase logic**. The ghost moves greedily to cut down its Manhattan distance to Pac-Man, so the agents learned to put a wall between themselves and the ghost, or to lure it into a loop in one half of the maze while they cleared pellets in the other half. That's how they survived despite the ghost's 2× speed.

The full write-up, with the MDP specification, training curves and hyperparameter analysis, is in the **[project report (PDF)](Report/neurips_2024.pdf)**. The LaTeX source is in [`Report/`](Report/).

## The environment

`PacManWorld` (registered as `pacman_env/PacManWorld`) is a 15×15 grid maze with one Pac-Man and one ghost.

- **Actions:** `right`, `up`, `left`, `down`. If the chosen move would hit a wall, Pac-Man keeps going in his current direction, as in the arcade game.
- **Observation:** the full grid (empty, Pac-Man, pellet, wall, ghost) plus Pac-Man's current direction.
- **Ghost:** moves 2 cells per step. Like the arcade ghosts, it never turns back unless it reaches a dead end. At each intersection it picks the direction that most reduces its distance to Pac-Man.
- **Pellet regeneration:** when fewer than 80% of the starting pellets remain, 10% of the empty cells are refilled at random.
- **Rewards:**
  - +1 per pellet
  - −5 for getting caught
  - −0.1 for each step, to discourage stalling
  - a growing penalty for staying on the same cell
  - a small shaping bonus for moving toward the nearest pellet
- **Episode end:** the episode terminates when the ghost catches Pac-Man. It is truncated, with a −1 penalty, after 100 steps without eating a pellet.
- **Rendering:** a simple pygame view, enabled with `render_mode="human"`.

### State encoding

Using the full grid as the state makes the Q-table far too large, so both agents share a compact encoding (`encode_state` in [`qpolicy.py`](qpolicy.py)):

```
(pacman_row, pacman_col, direction_to_nearest_pellet, ghost_distance_bucket)
```

`ghost_distance_bucket` sorts the Manhattan distance to the ghost into one of four bands: ≤1, ≤2, ≤4 or farther.

## Getting started

Requires Python 3 and pip.

```bash
git clone https://github.com/AshokSivathayalan/Pac-Man-RL.git
```

```bash
cd Pac-Man-RL
```

```bash
pip install -e .
```

### Watch a trained agent

This loads the saved Q-Learning table from `models/qlearning.pkl` and plays one episode in a pygame window:

```bash
python main.py
```

### Train

```bash
python qlearning.py   # 3,000 episodes, writes runs/qlearning/train.csv and models/qlearning.pkl
```

```bash
python sarsa.py       # hyperparameter sweep over γ, α and ε_start, logs to runs/sarsa/
```

Both scripts use ε-greedy exploration with exponential decay, `ε = max(ε_end, ε_start · e^(−episode/τ))`, and optimistic Q-value initialization. Every 100 episodes they run a 10-episode greedy evaluation and log it to CSV. Hyperparameters are set as constants at the top of each file.

## Project structure

```
pacman_env/
  envs/pacman_world.py   # The Pac-Man Gymnasium environment
  envs/grid_world.py     # Gymnasium's GridWorld tutorial env (starting template)
  wrappers/              # Example Gymnasium wrappers (from the template)
qpolicy.py               # Shared state encoding + Q-table loading
qlearning.py             # Q-Learning training loop
sarsa.py                 # SARSA training loop + hyperparameter sweep
main.py                  # Render a trained agent playing
models/                  # Saved Q-tables (pickled dicts)
runs/                    # Training/evaluation logs (CSV)
Report/                  # Final report (PDF + LaTeX source, NeurIPS template)
Biweekly_checkins/       # Course progress updates
```

## Team

Ashok Sivathayalan, Anna Vrbaski, Abaseen Ahmadzai and Shuva Gautam.

I built the Gymnasium environment: grid and step logic, ghost movement and capture, and pellet regeneration. I also implemented the SARSA training and ran its hyperparameter tuning.

## License

MIT. See [LICENSE](LICENSE).
