# Phase 1 build log

I built Phase 1 overnight, one commit per step. Read each entry here next to its commit
(`git log --reverse`, then `git show <sha>`). Each entry says what changed, why, the idea behind
it, and what to look at.

---

## Step 0: Setup

**What:** I brought back the uv / pytest / ruff tooling from the Sep 26 skeleton and switched to
the flat layout from your plan doc (`sim/`, `rankers/`, `experiments/`, `results/` at the repo
root). The old `src/dating_market_sim` package is gone. I also added numpy, pandas and matplotlib.

**Why flat:** your plan doc imports `sim.config` and `rankers.elo`, not
`dating_market_sim.sim.config`. Shorter imports make the experiment scripts easier to read. uv's
build backend handles a flat repo with two top-level packages:

```toml
[tool.uv.build-backend]
module-root = ""                  # packages live at the repo root, not under src/
module-name = ["sim", "rankers"]  # two importable top-level packages
```

`uv sync` installs both as editable, so `uv run python experiments/01_baselines_sweep.py` can
`import sim` from any directory.

**Also in this commit:** your move of the plan doc. The old `docs/` brainstorm is replaced by
`dating-market-simulator.md` at the root, and git history still has the brainstorm.

**Look at:** `pyproject.toml`.

---

## Step 1: Config and population

**What:** `sim/config.py` holds every number. `sim/population.py` generates one side of the
market. Tests: `tests/test_population.py`.

**Config design:**
- **Frozen dataclasses.** `SideConfig` holds the per-side numbers (w, like-rate Beta, trait
  importances, attention). `Config` holds everything else.
- **Variations:** `dataclasses.replace` makes them. `with_w(cfg, 0.8)` is one point of the w
  sweep, and `symmetric(cfg)` copies side A onto side B.
- **Presets:** `classroom(n)` and `hinge_stages(n)`. The stages add the Hinge mechanics in order:
  classroom → +scrolling → +newest_first → +carry_over.
- **Days come from an exposure target, not a fixed count.** Your doc's two examples imply
  different exposure: 12 days at 500 is 24% of the other side, while 30 days at 2,000 is 15%. I
  hold the share fixed, so a scale check compares like with like. Classroom: 0.24 → 12 days at
  500, 48 at 2,000. Hinge: 0.30 at ~20 views/day → 7 days at 500, 30 at 2,000.
- **Importance scale:** "moderate" = 1.0, "high" = 2.0, "high, slightly lower" = 1.8, and
  "moderate, somewhat higher" = 1.3.

**Population, the three ideas worth knowing:**
1. **Draw the noise, not the value.** `w = clip(mean + 0.15 · U[-1, 1])`, with the U[-1, 1]
   offset drawn once per person. Sweeping the w mean from 0.1 to 0.9 then moves *the same
   people* up together, so the sweep curves are smooth and differences come from w, not from
   re-shuffled people. `test_w_sweep_moves_the_same_people` checks this.
2. **A rank transform gives a uniform trait with a chosen correlation.** Fitness is
   `rank(0.4·u + √0.84·noise) / n`. The latent has corr exactly 0.4 with u, and ranking keeps
   that while making the marginal uniform on [0, 1] like the other traits. It needs no scipy.
3. **Separate RNG streams** (wired up in step 2). The population takes a `Generator` instead of
   a seed, so the world builder can give each random thing its own stream. Changing one part of
   the model then doesn't change every other random draw.

**What the numbers came out as (5,000 per side):**

| | like-rate mean | like-rate SD | corr(u, fitness) | corr(u, like-rate) |
|---|---|---|---|---|
| A | 0.404 | 0.203 | 0.40 | −0.24 |
| B | 0.105 | 0.064 | 0.40 | **−0.70** |

**Something to notice:** on side B, attractiveness explains most of the variation in
pickiness. Beta(4, 36) has an SD of only ~0.047, while the u-shift (−0.05 per SD of u) adds
about the same amount again. So on B, "picky" mostly means "attractive". That follows from your
placeholder numbers, not from a bug. Calibration might change it, but it's worth knowing when
reading the charts.

**Look at:** `make_side` in `sim/population.py`.
