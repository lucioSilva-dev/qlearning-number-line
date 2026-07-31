# Q-Learning on a Number Line

The smallest reinforcement-learning problem that still contains every moving part
of Q-learning: an agent stands on a line of integers `0 … 10`, can step left,
step right, or stop, and has to work out on its own that the only good place to
stop is at the far end.

No environment library, no neural network, no `numpy` — the Q-table is a plain
list of lists and the update rule is four lines of arithmetic.

```
0123456789 10      ← the agent (red) walks right until it reaches the goal,
     ^               then chooses to stop
```

## The problem

| | |
|---|---|
| **States** | the 11 integers `0 … 10` — the agent's position |
| **Actions** | `0` = step left, `1` = step right, `2` = stop |
| **Goal** | position `10` (`n` if `n` is even, otherwise `n-1`) |
| **Episode ends** | when the agent chooses `stop` — *not* when it reaches the goal |

That last row is the interesting design choice. Reaching the goal does not end
the episode; **deciding to stop** does. The agent therefore has to learn two
separate things: how to get to the goal, and when to declare that it has
arrived.

## Rewards

| Situation | Reward |
|---|---|
| Step left or right | `-1` |
| Stop anywhere | `-50` |
| Stop **on the goal** | `+100` |
| Step **onto** the goal | `+50` |
| Step into a wall (off either end) | `-25`, position unchanged |

The `-1` per step is what makes the agent prefer short routes: every wasted move
costs it. The `-50` for stopping is what stops it from quitting immediately —
without that penalty, "stop right now" would be the cheapest way to end an
episode with no further cost. Only on the goal does stopping flip from the worst
action to the best one, and the gap between `-50` and `+100` is what carves the
policy out.

## The update

```python
def qeq(s1, a, r, g, s2):
    return s1 + a*(r + g*s2 - s1)
```

This is the Bellman update in its plainest form: `Q ← Q + α (r + γ·max Q' − Q)`.

- `s1` — the current estimate `Q(state, action)`
- `r` — the reward just received
- `s2` — `max Q(next_state, ·)`, the value of the best action available next
- `g` (γ) — discount: how much the future is worth relative to the present
- `a` (α) — learning rate: how far the estimate moves toward the new evidence

The bracketed term `r + γ·max Q' − Q` is the **temporal-difference error** — the
gap between what the agent expected and what it actually saw. The whole
algorithm is just "nudge the estimate a fraction `α` in the direction of that
surprise", repeated until the surprises run out.

## How the value propagates

The trained table shipped in the file shows the mechanism clearly. Reading the
"step right" column from the goal backwards:

| State | Q(step right) |
|---|---|
| 9 | 706.88 |
| 8 | 585.04 |
| 7 | 470.06 |
| 6 | 360.80 |
| 5 | 256.83 |
| 4 | 160.40 |
| 3 | 78.97 |
| 2 | 23.25 |
| 1 | −0.52 |

The reward exists only at position 10, yet every state knows how good it is to
walk right. That is `γ·max Q'` doing its work: value leaks one step backwards per
sweep, from the goal outward, decaying as it goes. The gradient it leaves behind
*is* the policy — from any starting position, "pick the highest number" walks you
to the goal.

At state 10 the table inverts: `stop` is worth `780.24` against `0.0` for both
moves. The agent has learned where the destination is.

## Running

```bash
pip install colorama
python simple_rl.py
```

Out of the box this runs `test(qtable, 3)`: the agent starts at position 3 with
the pre-trained table and is animated walking to the goal, one step every two
seconds. `test` is purely greedy — it reads the table and never writes to it.

To train a fresh table instead:

```python
n = 10
qtable = [[0.0, 0.0, 0.0] for _ in range(n + 1)]
train(state=0, qtable=qtable, a=0.1, g=0.9, n=n, e=200)
```

| Parameter | Meaning |
|---|---|
| `state` | starting position of every episode |
| `qtable` | table to train, modified in place |
| `a` | learning rate (α) |
| `g` | discount factor (γ) |
| `n` | length of the line |
| `e` | number of episodes |

## Implementation notes

A few details in the code are worth calling out, because they explain behaviour
that would otherwise look surprising:

- **Training is greedy, with no exploration.** `train` always takes
  `getmax(qtable[s])`; there is no ε-greedy branch. Learning still works here
  because the initial table is all zeros and every action carries a negative
  reward: whichever action is tried first drops below the untried ones, which
  pushes the agent to try those instead. This "optimism under zero
  initialisation" is enough exploration for a line, but it does not generalise —
  in a problem with positive rewards along a wrong path, a greedy agent locks
  onto the first thing that works.
- **`getmax` breaks ties toward the highest index**, because it compares with
  `>=`. On an all-zero table that means `stop` wins the first comparison, which
  is precisely the action the `-50` penalty then teaches the agent to abandon.
- **`show()` sleeps for two seconds**, and `train` calls it on every step. That
  is deliberate for watching a single episode play out, but it makes real
  training runs impractically slow — comment the `show`/`print` calls out of
  `train` before running many episodes.
- **Walls do not end the episode.** Stepping off the end costs `-25` and leaves
  the agent where it was, so the boundary is a penalty to be learned around
  rather than a failure state.
- **`getmax` hard-codes `range(0,3)`**, so the action space is fixed at three.

## Files

| File | Purpose |
|------|---------|
| `simple_rl.py` | Environment, Q-learning update, training loop, greedy playback, and a pre-trained table |

## Where this goes next

The same three functions scale directly to a grid instead of a line — the only
change is that `moves` gains two entries and the state becomes a pair. Past that,
the table itself becomes the bottleneck: a Q-table needs one row per state, so
anything with a large or continuous state space needs a function approximator in
its place, which is the step from Q-learning to **deep Q-learning**.
