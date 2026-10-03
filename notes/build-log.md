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
