# Games

The games are playable environments that sit beside Key-to-Door. Each one sets up a different credit problem and has a knob that makes it harder, so a method can be pushed until it fails. They have no oracle, so they are scored on what the agent learns to do, not on its credit directly.

They were added on 2026-11-16, after the specification was frozen, and are not part of its train and held-out split.

## The games

| Game | Id | Credit problem | Knobs | Score |
|---|---|---|---|---|
| Key-to-Door | `KeyToDoor-v0` | One early action, one late payoff | `distractor_steps`, `apple_reward_std` | `has_key` |
| Coin or Key | `CoinOrKey-v0` | A reward now against a larger one later | `coin_reward`, `distractor_steps` | `has_key` |
| Lottery Keys | `LotteryKeys-v0` | An effect that only shows up on average | `bronze_prob`, `distractor_steps` | `best_key` |
| Many Keys | `ManyKeys-v0` | Several actions sharing one payoff | `num_keys`, `need_all` | `keys_held` |
| Decoy | `Decoy-v0` | Something that goes with reward without causing it | `distractor_steps`, `charm_cost` | `took_charm` (lower is better) |
| Delayed Catch | `DelayedCatch-v0` | Many similar decisions paid as one lump | `payout_every`, `observe_score` | `catch_rate` |
| Delayed Payout | `DelayedPayout-v0` | The same, on a standard control game | `payout_every`, `base_id` | `mean_return` |

Scores are the `info` values at the end of an episode, which the sweep averages and reports without the `info_` prefix.

## Rules

### Grid games

Coin or Key, Lottery Keys, Many Keys and Decoy ([src/games/grid.py](../src/games/grid.py)) use Key-to-Door's layout and take all of its parameters. The agent moves on a 5 × 5 grid through a key phase (10 steps), an apple phase (`distractor_steps`, each apple pays 1 and respawns) and a door phase (10 steps). Positions are redrawn at the start of each phase. The observation shows the agent, the objects, the phase, the time left in it and what the agent holds. With `observe_key` off, what it holds is hidden.

**Coin or Key.** The key phase holds a key and a coin, and touching one removes the other. The coin pays `coin_reward` at once. The key pays `door_reward` (10) when the episode ends. There is no door to walk to: with one, the key would be worth less than the coin until the agent had learned to reach doors, and correct credit would then favour the coin.

**Lottery Keys.** The key phase holds a gold and a bronze key, and touching one removes the other. Reaching the door ends the episode and opens it with probability `gold_prob` (0.9) or `bronze_prob`.

**Many Keys.** The key phase and the apple phase repeat `num_keys` times before the door phase. The door pays `door_reward` for each key held, so with one key the game is Key-to-Door. With `need_all` it pays only when every key is held.

**Decoy.** With probability `unlocked_prob` (0.5) the door starts unlocked and pays without a key. In those episodes only, a charm lies beside the key. Taking it costs `charm_cost` and changes nothing else. The charm is the only sign that the door is unlocked: it shows on the grid during the key phase and, once taken, as a flag for the rest of the episode. Holding it therefore predicts the payoff without causing it, and the true credit for taking it is `−charm_cost`. With `observe_luck` the observation says whether the door is unlocked throughout, which removes the trap. `took_charm` is a share of all episodes, so it cannot exceed `unlocked_prob`; random play gives about 0.11. `correct` is 1 when the agent left the charm (door unlocked) or took the key (door locked).

### Delayed Catch

A ball falls down a 5 × 5 board, one row per step, and the paddle on the bottom row moves left, stays or moves right ([src/games/catch.py](../src/games/catch.py)). A round lasts 4 steps and an episode 40 rounds. Each catch is worth 1, paid as a sum every `payout_every` rounds. The observation shows the unpaid catches unless `observe_score` is off.

### Delayed Payout

A wrapper around any Gymnasium game ([src/games/delayed.py](../src/games/delayed.py)). It withholds the game's rewards and pays their sum every `payout_every` steps, or only at the end when `payout_every` is 0. The time limit counts as the end of the episode. The base game is `CartPole-v1`; set `base_id` for another. `LunarLander-v3` needs Box2D, which does not build on Python 3.14 here.

## Ladders and limits

Each game has an experiment file in [experiments/](../experiments/) with the four baselines, 10 seeds and 1,000,000 steps per run (500,000 for Delayed Payout), the same learner settings as Key-to-Door.

The file's `ladders` list orders some of its tasks from easiest to hardest:

```json
{"name": "delay", "score": "has_key", "pass": 0.8, "levels": [5, 2, 6, 7]}
```

`levels` are positions in the file's `env_kwargs` list. A method passes a rung when its score, averaged over seeds at the end of training, is at least `pass` (at most, with `"lower": true`). Its **limit** is the last rung it reaches without failing an earlier one. The first rung of every ladder is the game without the difficulty: a method that fails there is failing at the game, and its limit is `none`.

```
.venv/bin/python -m src.sweep experiments/coin_or_key.json --workers 10
```

prints one table per ladder, with the score at every rung and the limit, and saves them to `limits.csv` in the run folder.

| File | Ladders |
|---|---|
| `coin_or_key.json` | Coin value 0 to 9 at delay 30; delay 0 to 300 at coin value 5 |
| `lottery_keys.json` | Bronze key 0 to 0.79 against gold 0.9, at delay 30; delay 0 to 300 at bronze 0.45 |
| `many_keys.json` | 1, 2, 4, 8 keys; 2 keys with all needed |
| `decoy.json` | Delay 0 to 300 at charm cost 1; charm cost 4 down to 0.25 at delay 30; luck visible against hidden |
| `delayed_catch.json` | Payout every 1 to 40 rounds, with the score visible and hidden |
| `delayed_payout.json` | Payout every 1, 5, 20, 100 steps and at the end, on CartPole |

## Limits of the games themselves

- The agent has no memory, so every game can be played from the current observation. A game that needs the agent to remember a cue would fail for every method.
- Exploration is kept out as far as possible: random play reaches each payoff often. The exception is Many Keys with `need_all`, where random play opens the door in under 1% of episodes with two keys. More keys than that would test exploration.
- Without an oracle, a failure on a hard rung cannot be split into missing credit and a hard game. The first rung is the only control.
