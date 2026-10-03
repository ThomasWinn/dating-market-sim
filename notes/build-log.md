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

---

## Step 2: True preferences (P, Q) and the World ⚠️ includes a weight change, please read

**What:** `sim/preferences.py` builds `P[a, b]` (chance A likes B) and `Q[b, a]` (chance B
likes A) once per world. `World` bundles the people, P, Q, and the pre-drawn decisions. Tests:
`tests/test_preferences.py`.

### Ideas worth knowing
1. **Broadcasting instead of loops.** `taste` is an (n_viewers × n_targets) matrix built by
   looping over the 5 traits only. For example, `viewers.ideals[:, j, None] - targets.traits[None, :, k]`
   is (n, 1) minus (1, n), which numpy broadcasts to (n, n). That's every viewer–target pair
   in one vectorized line.
2. **Why standardize before mixing with w.** `score = w·z(u) + (1−w)·z_row(taste)`. If you
   mixed raw values, whichever part had the bigger spread would dominate whatever w was. After
   z-scoring, both parts have SD 1, so w = 0.6 really means "60% appeal".
3. **Threshold calibration (choice #2 in the plan).** Your doc's quantile threshold gives:

   | | target like-rate | quantile cut, realized |
   |---|---|---|
   | A | 0.406 | 0.418 |
   | B | 0.103 | **0.145** (+40%) |

   The reason: the sigmoid gives people *just below* the cut a real chance too, and a picky
   person has many more people just below the cut than above it. `calibrated_thresholds`
   bisects every row at once (40 halvings) until mean(P row) equals the target exactly.
   `test_quantile_threshold_overshoots_low_like_rates` keeps the "before" as a documented fact.
4. **Pre-drawn decisions (choice #1).** `DA = U < P` is drawn once, so A's answer about B is
   fixed for the whole run. Two payoffs: repeat looks are consistent, and every ranker faces
   the same coin flips (*common random numbers*, a standard variance-reduction trick in
   simulation and A/B-test analysis).
5. **Named RNG streams.** `rng_for(seed, "chemistry")` and friends: each random part has its
   own stream from `SeedSequence.spawn`. In the w sweep, the same seed gives the same people,
   chemistry, and decision coin flips. Only w changes, so the curves isolate w's effect.

### ⚠️ The sanity check failed at first, and I changed your fitness/ambition weights
Your sanity check, "high w → likes concentrate heavily; low w → much flatter", **failed** with
the plan's importances. Expected likes received:

| w (side A) | Gini A→B | Gini B→A | likes track... |
|---|---|---|---|
| 0.1 (plan weights) | 0.38 | 0.70 | fitness (corr 0.88) |
| 0.9 (plan weights) | 0.41 | 0.72 | u (corr 0.96) |

w only changed *who* got the likes, not how unequal the split was.

**Why:** I measured each trait term's spread within a viewer's row. Fitness alone was 63% of
taste variance, and fitness plus ambition together 78%. Two things combine:
- Everyone ranks more-is-better traits the same way, so they are universal, just like `u`.
- The ideal-point terms barely spread. Ideals cluster toward the middle (ρ = 0.5), and
  closeness `1 − |ideal − trait|` has a squashed range, giving a variance of about 0.04 each.
  Fitness, at importance 2.0, has a variance of 0.34.

So "type" was mostly a second universal score. Your doc predicted exactly this: *"If charts
show extreme concentration even at low w, check the fitness weights first."*

**What I did:** I kept your pattern (fitness weighted more than ambition; B a bit lower on
fitness and higher on ambition) and scaled the two more-is-better weights down 4×. A:
fitness 2.0 → 0.5, ambition 1.0 → 0.25. B: 1.8 → 0.45, 1.3 → 0.325. The universal share drops
to ~18%. The original values are in a comment in `sim/config.py`, so reverting is a two-line
change. The sweep now behaves:

| w (side A) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| Gini likes received A→B | 0.27 | 0.29 | 0.33 | 0.38 | 0.42 |
| Gini likes received B→A | 0.49 | 0.51 | 0.57 | 0.65 | 0.71 |

**For you to decide:** the honest reading is that in your model, "fitness is high importance"
and "w controls how universal attraction is" can't both hold. Universality really comes from
w *plus* any trait everyone agrees on. Other fixes would be to standardize each trait term
(so importance means influence), or to treat fitness as part of appeal rather than type. I
went with the smallest change. Two new tests guard it: `test_taste_is_mostly_personal`
(universal share < 30%) and `test_high_w_concentrates_likes_on_high_u` (Gini rises ≥ 0.1).
**Interview angle:** "the knob I thought controlled universality wasn't the only source of
it" is a good story about checking your assumptions with a measurement.

**Speed:** `build_world` takes 0.07 s at 500 per side and 1.1 s at 2,000.

**Look at:** `scores` and `calibrated_thresholds` in `sim/preferences.py`.

---

## Step 3: Metric helpers

**What:** `gini` and `by_decile` in `sim/metrics.py`, with hand-checkable tests in
`tests/test_metrics.py`. The full per-run summary (matches, zero-match %, dead likes, and so on)
comes in step 4, because it needs the simulator's output.

**Gini, in one line:** sort the values, then weight each by its rank position:
`Σ (2i − n − 1)·x_i / (n · Σx)`. Two checks worth remembering (both are tests):
- If everyone has the same amount, Gini is 0.
- If one person has everything, Gini is (n − 1)/n, which approaches 1 as n grows.

For dating markets, Gini of *likes received* is the headline inequality number people quote.
Gini of *matches* is the one that matters to users.

**by_decile:** sort people by a key (here, hidden appeal `u`), split them into ten equal
groups, and average something per group. This is the "who gets left behind" chart.

**Look at:** `tests/test_metrics.py`. Each test is a tiny worked example.

---

## Step 4: The market loop (classroom mode) and the random ranker

**What:**
- `sim/market.py`: the daily loop and the event log.
- `rankers/base.py`: shared list-building code that every ranker uses.
- `rankers/baselines.py`: `RandomRanker`.
- `summarize()` in `sim/metrics.py`: one row of metrics per run.
- Tests: `tests/test_market.py`, 14 invariants.

### How a day works (`Market.run`)
1. **`review(day)`:** each side's inbox is three flat arrays (`recipient`, `sender`,
   `day_sent`), one entry per pending like. The vectorized "top-k per group" trick:
   ```python
   order = np.lexsort((random_tiebreak, priority, recipient))  # last key sorts first
   rec = recipient[order]
   rank_in_group = np.arange(len(rec)) - np.searchsorted(rec, rec)  # 0, 1, 2... per recipient
   take = rank_in_group < attention[rec]
   ```
   `searchsorted(rec, rec)` returns where each recipient's group starts, so subtracting it
   gives each like's rank within its recipient's queue. It's worth learning: "top-k within
   each group without a Python loop" comes up everywhere in ranking code.
2. **`browse(day)`:** eligible = not seen and not waiting in my inbox. The ranker turns that
   into an ordered list. The swiping itself is also vectorized:
   - `cumsum` of would-like along the list shows where the like cap is hit.
   - `geometric(quit_prob)` gives each person's patience.
   - Profiles viewed = min(cap stop, patience, list length). Classroom mode is the same code
     with `quit_prob = 0` and `list_len = 10`, so Hinge-mode scrolling needs no second code
     path.
3. **`end_of_day`:** in classroom mode, unreviewed likes die. Then today's likes are delivered
   for tomorrow.

After the last day there is one review-only round, so the last day's likes get a fair chance.

### Rankers share one list builder (`Ranker.lists`)
A ranker only provides `scores(side, state)`, an (n_viewers × n_candidates) matrix. The base
class does the rest:
- masks ineligible candidates to −∞
- picks the top k with `argpartition`, which is O(n) per row instead of a full sort
- fills exploration slots if the ranker uses them
- pushes −1 padding to the end

Exploration slots are spread through the list (every 10th position) rather than tacked on
the end, so a Hinge scroller who quits after ~20 profiles still reaches them.

### Bookkeeping choices
- **Every like is logged exactly once** in the inbox log: reviewed (with `day_reviewed`),
  died (`day_reviewed = -1`), or superseded. A test checks inbox rows == browse likes.
- **A "dead like"** = never reviewed *and* the pair never matched. If A and B like each other
  on the same day and one like-back happens first, the other like is resolved, not dead.
- **Morning state:** both sides' rankers see the same morning state, because new likes are
  applied only after both sides have browsed. Without this, side A's swipes would leak into
  side B's ranking on the same day.

### I changed the classroom exposure target from 0.24 to 0.22 (11 days at 500, not 12)
Reviewing someone's like in your inbox also counts as seeing them, so it also shrinks your
pool. Side B gets a lot of incoming likes from the like-happy side A. At 12 days B saw 32% of
side A, which is over your ~30% guideline. At 11 days: A 24%, B 29%, at both 500 and 2,000 per
side (the exposure-derived days keep it steady). A new test enforces ≤ 30%.

### Numbers (random ranker, classroom mode, seed 0)
| n/side | days | runtime | matches | zero-match % | dead likes to A / to B | like-rate A / B |
|---|---|---|---|---|---|---|
| 500 | 12* | 0.07 s | 3,209 | 14.7% | 1.2% / 5.8% | 0.395 / 0.100 |
| 2,000 | 48* | 3.9 s | 55,010 | 4.4% | 1.0% / 6.3% | 0.404 / 0.102 |

\*Measured before the exposure change. The realized like-rates match targets (0.406 / 0.103),
which confirms the calibration end to end. Likes *to B* die ~5× more often than likes *to A*:
the picky side is also the side whose inbox overflows.

**Look at:** `Market.review` and `Market.browse`, then `Ranker.lists`.

---

## Step 5: Popularity, Elo, and the two oracles

**What:**
- `rankers/baselines.py`: `PopularityRanker`, `OneSidedOracle`, `ReciprocalOracle`.
- `rankers/elo.py`: `EloRanker`.
- `sim/preferences.py`: the side constants (`SIDES`, `OTHER`) moved here, next to `World`, so
  rankers can use them without a circular import. Also adds `World.probs(side)`.
- Tests: `tests/test_rankers.py`, 12 tests, including a hand-computed Elo update.

### The rankers in one line each
- **Popularity:** likes received + U[0, 1) noise. Counts are integers, so the noise only breaks
  ties, and it's drawn fresh per viewer so day 1 isn't the same list for everyone. Every 10th
  slot is exploration.
- **Elo:** each swipe is a game that the *person swiped on* wins (a like) or loses (a pass),
  against the swiper's rating. Hand-checked in a test: a 1,600 swiper liking a 1,400 target
  gives +24.3, and a pass gives −7.7. A day's swipes are batched from the morning ratings
  (`np.add.at` handles repeated targets). The ranking shows people at a similar
  **percentile within their own side**, because raw ratings aren't comparable across sides.
- **One-sided oracle:** P (how much I'd like them).
- **Reciprocal oracle:** P ∘ Qᵀ (the true match chance).

### Results (classroom, 500/side, default w, mean of 3 seeds)
| ranker | matches | zero-match % | Gini matches | dead likes to A / B | like-rate A / B |
|---|---|---|---|---|---|
| popularity | 2,052 | 36.2 | 0.73 | **81% / 86%** | 0.66 / 0.27 |
| elo | 2,554 | 21.2 | 0.61 | 0.2% / 4.4% | 0.36 / 0.07 |
| random | 3,072 | 14.8 | 0.56 | 1.3% / 6.1% | 0.40 / 0.10 |
| one-sided oracle | 5,422 | 13.0 | 0.58 | 35% / 56% | 0.81 / 0.36 |
| reciprocal oracle | **6,702** | **9.2** | 0.56 | 19% / 18% | 0.58 / 0.28 |

Your sanity checks pass: reciprocal ≥ everything, and reciprocal > one-sided.

### Three things worth understanding
1. **Popularity is *worse* than random.** It herds everyone toward the same few people, and
   81–86% of likes then die unread in their overflowing inboxes. Likes are high (0.66), but
   matches are low. This is the congestion story the MODE paper (Phase 2) is about, and here
   it shows up in a heuristic real apps have used.
2. **The one-sided oracle has the same flaw, milder.** It shows you who *you* like most,
   which means the universally appealing people, so their inboxes clog (35% / 56% dead).
   Adding "will they like me back" (reciprocal) roughly halves the dead likes and adds 24%
   matches. That's the whole case for reciprocal recommendation in one row.
3. **Elo is below random, and it isn't a bug.** I checked, because by the rearrangement
   inequality, pairing similar-appeal people *should* raise expected matches:
   - Elo's ratings do track hidden appeal (corr 0.71 on A, 0.87 on B).
   - The pairs it shows *do* have a higher match chance (mean P·Q 0.0344 vs 0.0293).
   - **But percentile neighbourhoods are symmetric.** If A sits near B's percentile, then A
     sees B *and* B sees A. With decisions drawn once, the second look adds nothing. Only 65%
     of Elo's views are distinct pairs, vs 91% for random (38,717 pairs seen from both
     sides). 17% better pairs × 29% fewer distinct pairs = fewer matches.

   **A useful identity falls out of this:** matches ≈ Σ P·Q over distinct pairs evaluated
   (random: 3,031 predicted vs 2,968 actual; Elo: 2,521 vs 2,480). The reciprocal oracle
   loses ~8% to congestion (7,086 → 6,509), and that gap is exactly what a market-level ranker
   like MODE tries to win back.

**Interview angle for #3:** "the per-impression metric went up, but the system metric went
down, because impressions weren't independent." That's the offline-vs-online gap in
miniature.

**Look at:** `rankers/elo.py` (short), then `ReciprocalOracle` in `rankers/baselines.py`.

---

## Step 6: The classroom sweep (`experiments/01_baselines_sweep.py`)

**What:**
- `sim/runner.py`: a parallel sweep runner. Each (config, seed) world is built once, every
  ranker runs on it, and `ProcessPoolExecutor` spreads the work over 14 cores.
- `sim/plots.py`: shared chart style. Colors follow the ranker, in a fixed palette order
  checked with a colorblind-safety validator.
- `experiments/01_baselines_sweep.py`: 5 rankers × 9 w values × 3 seeds = 135 runs.
  **It takes 2 seconds at 500 per side.**

Run it:
```bash
uv run python experiments/01_baselines_sweep.py --mode classroom          # 500/side
uv run python experiments/01_baselines_sweep.py --mode classroom --n 2000 # final numbers
```

Outputs in `results/`, each with the CSV twin `results/n500/01_classroom_sweep.csv`:
`01_classroom_matches_vs_w.png`, `_gini_vs_w.png`, `_dead_likes_vs_w.png`,
`_u_deciles.png`, `_oracles.png`.

**A plan change:** I commit `results/` instead of gitignoring it. The README and the HTML
report both show these charts, and they're small. They regenerate exactly, since seeds are
fixed.

### Self-check against your sanity list (all pass)
| check | result |
|---|---|
| High w concentrates likes, low w is flatter | Gini of likes received by B (random): 0.29 at w = 0.1 → 0.42 at 0.9. Gini of matches: 0.48 → 0.66 |
| Reciprocal oracle ≥ random, popularity, Elo | yes, at every w |
| Reciprocal > one-sided oracle | yes, at every w (e.g. 6,702 vs 5,422 at w = 0.6) |
| Realized like-rates ≈ targets | 0.395 / 0.100 vs 0.406 / 0.103 (step 4) |
| Exposure ≤ 30% | max 29.9% (one-sided oracle, side B, w = 0.1) |

### What the charts say
- **Matches fall as attraction gets more universal, for every ranker.** At w = 0.1 the
  reciprocal oracle makes 11.0k matches; at 0.9 it makes 4.9k. When everyone wants the same
  people, most likes go to people who won't like back.
- **Popularity is the worst at every w:** about 80–86% of its likes die unread. It creates its
  own concentration through a feedback loop (liked → shown more → liked more), so it's bad
  even when tastes are personal (w = 0.1).
- **"Who gets left behind" (w = 0.8):** on side A the bottom half of hidden appeal gets
  almost nothing under *every* ranker. On side B under random, matches *peak at decile 8
  and then fall*. The most attractive B's are also the pickiest (the −0.70 correlation from
  step 1), so they like back less.
- **One-sided vs. reciprocal:** at w = 0.9 the one-sided oracle lets 67% of likes die, vs 30%
  for reciprocal, with 32.6% vs 24.8% of users at zero matches.

**Look at:** `results/n500/01_classroom_oracles.png` first. It's the one-picture argument for
reciprocal recommendation.

---

## Step 7: Hinge mode, one mechanic at a time (`experiments/01b_hinge_mode.py`)

**What:**
- `tests/test_hinge.py`: one test per mechanic, plus the emergent-behaviour tests.
- A new metric, `cap_hit_share`: the share of browse days that ended at the like cap.
- `experiments/01b_hinge_mode.py`: 5 rankers × 4 cumulative stages × 3 seeds, in 1 second.

The code paths themselves landed in step 4, because each one is a single branch.
`inbox_order` picks the review sort key, and `carry_over` decides whether `end_of_day` drops
unreviewed likes. Scrolling is only numbers (`list_len = 50`, `quit_prob = 0.05`). This step
tests each mechanic in isolation and measures it.

**Tests that pin each mechanic down:**
- **best-first:** with 40 likes waiting, the reviewer handles exactly their top-`attention` by
  their own Q.
- **newest-first:** with 10 likes from each of days 0–3, day 3's go first.
- **carry-over:** unreviewed likes stay pending (or vanish without it), and in a full run some
  likes are reviewed more than a day late.
- **emergent behaviour:** side A's cap-hit share is more than 5× side B's, and side B views
  more profiles but sends less than half as many likes.

**Exposure, again:** the scrolling stage put side B at 30.4% (inbox reviews again). I lowered
the Hinge browse target from 0.30 to 0.27, which gives **6 days at 500 and 27 at 2,000**. A
test now checks every stage stays ≤ 30%.

### Results (500/side, 6 days, mean of 3 seeds)
| stage | random | popularity | elo | one-sided | reciprocal |
|---|---|---|---|---|---|
| classroom | 1,743 | 1,160 | 1,470 | 3,677 | 5,479 |
| + scrolling | 2,505 | 2,155 | 2,187 | 3,750 | 5,590 |
| + newest-first | 2,420 | 1,461 | 2,160 | 3,058 | 5,028 |
| + carry-over | 2,486 | 1,504 | 2,164 | 2,871 | **4,410** |

**1. Scrolling: your predicted emergent effect shows up, and nobody coded it.**
Random ranker:

| | profiles viewed/day | likes sent/day | days ending at the cap |
|---|---|---|---|
| A | 13.1 | 4.6 | **35%** |
| B | **18.1** | 1.8 | 3.8% |

Side A burns through its 8 likes. Side B scrolls further and quits from boredom. The like cap
is a real constraint for one side only. Product lesson: a "more likes" subscription is worth
far more to side A.

**2. Newest-first only hurts the rankers that congest inboxes.** Reciprocal: −10%, one-sided:
−18%, popularity: −32%, random and Elo: about flat. Without carry-over every waiting like
arrived the same day, so "newest-first" is really *random order*. Losing best-first only
matters when there are more likes than attention.

**3. Carry-over *lowers* matches for the reciprocal oracle (5,028 → 4,410), even though it
keeps likes alive. I measured why.** The count of mutual-but-never-matched pairs (both would
like each other, a like was sent, no match) rose from 1,517 to 2,145 (+628), which lines up
with the 560 lost matches. Here's the mechanism:
- Without carry-over, an unread like dies, and the pair can still meet later in browse.
- With carry-over, the like sits in a flooded inbox forever. Newest-first buries it, and
  because people waiting in your inbox are excluded from your feed, that pair can never meet
  another way.

**A stale like in a flooded inbox is worse than no like.** Under random ranking, inboxes
don't flood and carry-over slightly helps (119 → 88 missed pairs). This is a congestion
effect, and exactly the kind MODE reasons about.

**Look at:** `results/n500/01b_sides_by_stage.png`, then `results/n500/01b_outcomes_by_stage.png`.

---

## Step 8: Both modes from one command, final 2,000/side runs, scale check

**What:**
- `01_baselines_sweep.py` now runs **both modes by default**. That's your doc's "done when":
  one command produces the baseline sweep charts in both modes.
- Outputs go to `results/n<size>/`, so the 500/side development runs and the 2,000/side
  final runs sit side by side. Earlier log entries now point at `results/n500/`.
- `experiments/01c_scale_check.py` runs 500 / 1,000 / 2,000 per side in both modes.

```bash
uv run python experiments/01_baselines_sweep.py            # 500/side, both modes: 4 s
uv run python experiments/01_baselines_sweep.py --n 2000   # final: 1 min 41 s (14 cores)
uv run python experiments/01b_hinge_mode.py --n 2000       # 23 s
uv run python experiments/01c_scale_check.py               # 23 s
```

### Scale check: the ranker order is identical at every size, in both modes
| rank | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| all sizes, both modes | reciprocal | one-sided | random | Elo | popularity |

Per-user-per-day rates are roughly flat across sizes (`results/01c_*_scale.png`), which
confirms the exposure-based day count keeps sizes comparable. Gini of matches falls slightly
as the market grows (0.53 → 0.50 for reciprocal in Hinge mode): with more days, there's more
chance for everyone.

### Final numbers (2,000/side, default w = 0.6, mean of 3 seeds)
| | random | popularity | Elo | one-sided | reciprocal |
|---|---|---|---|---|---|
| classroom matches | 50,581 | 19,050 | 40,846 | 89,018 | **111,263** |
| hinge matches | 45,393 | 21,727 | 38,669 | 52,030 | **72,010** |
| classroom dead likes | 4.9% | **92.5%** | 6.0% | 48.8% | 18.3% |
| hinge dead likes | 6.0% | 77.2% | 1.8% | 55.0% | 24.6% |

**Hinge mode hurts the oracles most.** One-sided: −42%, reciprocal: −35%, random: −10%.
The rankers that send everyone to the same people suffer most when inboxes are reviewed
newest-first and stale likes block rediscovery (step 7).

### Two details, reported as they are
- **Exposure peaked at 30.3%** in one sweep config (side B, classroom). That's within your
  "~25–30%" but 0.3 points over 30%. I didn't shave the day count again for it.
- **Side A's realized like-rate in Hinge mode is 0.35, not 0.41.** That's not a calibration
  bug. The heaviest likers hit the cap after a few profiles and stop viewing, so they make up
  a smaller share of all views. *Per-view* metrics are biased toward whoever views the most,
  which is worth remembering when reading any app's "like-rate" dashboard. Classroom mode
  stays on target (0.397 / 0.103) because almost everyone reaches the end of their 10
  profiles.
