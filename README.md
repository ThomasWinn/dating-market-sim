# dating-market-sim

I've always wondered how dating apps decide who sees whom, and why so many matches just... go quiet.
This is a synthetic sandbox for poking at that: a toy Hinge-style market full of made-up people,
where I can swap ranking algorithms in and out and see what happens.

## The two questions

1. **Ranking:** who gets shown to whom? Elo, stable matching, a learned model: how do they change
   who matches, and how unequal things get?
2. **Ghosting:** once people match, what happens? My hunch is that a lot of ghosting comes from the
   market itself (too many matches, not enough attention) and not just from personalities clashing.

## Things I want to try

- [x] Random vs. Elo ranking: match rate and inequality (Gini)
- [x] Stable-matching-style "most compatible" picks
- [ ] A learned ranker that predicts *mutual* likes
- [x] The knob `w`: how much attraction is universal vs. individual taste
- [ ] Ghosting that emerges on its own from attention budgets and competing matches
- [ ] Side quests: new-user boosts, swipe-right-on-everyone users, how much better photos move you

## Ground rules

- Synthetic people only: no real profiles, messages, or photos.
- No images and no face-attractiveness model. Everything is numbers and traits.

## Status

Phase 1 (simulator + baseline rankers) is done. Phases 2-4 (MODE, matrix factorization, a
reciprocal two-tower) are planned in [`dating-market-simulator.md`](dating-market-simulator.md).
The build, step by step and with the reasoning, is in [`notes/build-log.md`](notes/build-log.md).

## Phase 1: the simulator and baselines

Two sides, A and B, with known true preferences: `P[a, b]` is the chance A likes B, and
`Q[b, a]` is the chance B likes A. Each person's like is a mix of general appeal `u` (hidden)
and fit with their own type (displayed traits), weighted by `w`. Side A likes often and leans
on looks; side B is picky. Every day, everyone reviews incoming likes up to an attention
budget, then browses a ranked list. A like-back is a match.

- **Classroom mode:** 10 profiles a day, best-first inbox, unreviewed likes vanish overnight.
- **Hinge mode:** scroll up to 50 profiles with an 8-like cap and a 5% chance of quitting
  after each profile, newest-first inbox, unreviewed likes carry over forever.

**Rankers:** random, popularity, Elo, a one-sided oracle (true P), a reciprocal oracle (true
P × Q), and Gale-Shapley as a daily top pick. The oracles read the true preferences, so
they're reference points, not something an app could run.

### Run it

```bash
uv sync
uv run pytest                                            # 91 tests, ~3 s
uv run python experiments/01_baselines_sweep.py          # both modes, 500/side: ~4 s
uv run python experiments/01_baselines_sweep.py --n 2000 # final numbers: ~2 min on 14 cores
uv run python experiments/01b_hinge_mode.py --n 2000     # add Hinge mechanics one at a time
uv run python experiments/01c_scale_check.py             # 500 / 1,000 / 2,000 per side
uv run python experiments/01d_top_pick.py                # Gale-Shapley top pick
```

Every number lives in `sim/config.py`. Charts and their CSV twins are written to `results/`.

### Charts

![Matches vs w](results/n2000/01_classroom_matches_vs_w.png)
![One-sided vs reciprocal oracle](results/n2000/01_classroom_oracles.png)
![What each Hinge mechanic does](results/n2000/01b_outcomes_by_stage.png)
![Side A vs side B behavior](results/n2000/01b_sides_by_stage.png)
![Who gets left behind](results/n2000/01_classroom_u_deciles.png)

### Three findings

1. **Asking "will they like me back?" is worth about 25% more matches and halves the dead
   likes.** At 2,000 per side, classroom mode, w = 0.6: the reciprocal oracle makes 111,263
   matches vs. 89,018 for the one-sided oracle, and 18% of its likes die unread vs. 49%. Showing
   people who *you* like most sends everyone to the same few inboxes.
2. **Congestion beats compatibility.** Popularity ranking does worse than random (19,050 vs.
   50,581 matches) because 92% of its likes die in overflowing inboxes. In Hinge mode, the
   rankers that herd lose the most (reciprocal −35%, one-sided −42%, random −10%). Carry-over
   plus newest-first even *locks out* mutual pairs: a like stuck unread in a flooded inbox also
   hides that pair from each other's feeds.
3. **Better pairs per impression can still mean fewer matches.** Elo and Gale-Shapley both show
   pairs with better match odds than random, but their neighbourhoods are symmetric (if A is
   shown B, B is shown A), so they cover fewer distinct pairs. Elo lands below random, and
   Gale-Shapley's top pick makes about a third as many matches as the reciprocal top-1. Across
   every ranker, matches ≈ the sum of P × Q over the distinct pairs evaluated, minus what
   congestion eats.

The behaviour the plan predicted emerges on its own in Hinge mode: side A ends 35% of its days
at the like cap, while side B scrolls further (18 vs. 13 profiles a day), quits from impatience,
and sends far fewer likes.

### Limitations

- Opposite-side matching only, with everyone active every day.
- Preferences are fixed for the whole run: nobody gets pickier as they get more likes.
- Every number is a placeholder until it's calibrated against real data (Libimseti, the
  Columbia speed-dating study). The fitness and ambition weights were scaled down 4× from the
  plan, so that `w` really controls how universal attraction is (see the build log, step 2).
- Each person's answer about each other person is drawn once, so a second look never changes
  a decision. That's realistic for "passed means passed", but there's no mood noise.
- Days run in synchronized rounds (everyone reviews, then everyone browses).

## Dev

```bash
uv sync
uv run pytest
uv run ruff check .
```
