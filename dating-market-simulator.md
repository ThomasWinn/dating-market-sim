# Dating Market Simulator — Implementation & Research Plan
*Revised Oct 4, 2026.*

## Core idea
A simulated two-sided dating market where **I know everyone's true preferences**. That makes it possible to:
- run any ranking algorithm and see how the market reacts (a static dataset can't do this)
- measure how well a learned model recovers the true preferences
- compare every algorithm against perfect-information reference points

The problem has two separate halves, built separately:
1. **Preference estimation:** who likes whom? (Elo, matrix factorization, reciprocal two-tower)
2. **Ranking / market design:** given those estimates, what does each person see? (naive, reciprocal product, TU, MODE)

---

## Phase overview

| Phase | Build | Read (only these parts) | Done when |
|---|---|---|---|
| 1 | Simulator (classroom mode, then Hinge mode) + random / popularity / Elo / one-sided & reciprocal oracles (+ Gale-Shapley stretch) | Elo explainer; Gale-Shapley on Wikipedia | One command produces the baseline sweep charts in both modes |
| 2 | Full reproduction of MODE's synthetic experiment, then MODE inside the simulator | MODE §2–3, §4.1, Appendix B | My Figure 1a looks like theirs |
| 3 | Matrix factorization per direction → estimated preferences | ALS tutorial | Estimated vs. true preference correlation reported |
| 4 | Hinge-style reciprocal two-tower in PyTorch | Hinge 2026 paper | Two-tower beats MF on cold start; oracle-vs-learned gap measured |

---

## Phase 1 — The simulator and baselines

### Structure
- **1a — Classroom mode:** simple, controlled rules (close to the MODE paper's world). Get every ranker working and every chart sensible here first.
- **1b — Hinge mode:** add realistic mechanics **one at a time**, re-running the baseline sweep after each, so every mechanic's effect is measured on its own.

### Config and scale
- **Every number lives in `sim/config.py`.** No magic numbers in code. Only build switches I'll actually flip (the Hinge-mode mechanics are the only alternative code paths).
- **Asymmetric vs. symmetric is not a switch:** it's just per-side numbers. Setting both sides' numbers equal gives the symmetric market for free.
- **Users per side is a parameter.** Develop at **500 per side** (fast iteration); run final experiments at **2,000 per side**.
- **Cap exposure, not just days:** each user should see at most ~25–30% of the other side over a full run (prevents pool exhaustion from compressing differences between rankers). Classroom mode at 2,000/side: 10 profiles × 30 days = 15%. At 500/side, use ~12 days.
- Fixed random seeds everywhere. Every result run with 3–5 seeds, reported as mean ± spread.
- **All default numbers below are placeholders.** Calibration against real data is in the backlog.

### The user model

**Sides:** A and B, opposite-side matching only for now (limitation, noted). Default is asymmetric: side A likes more often and weights looks more; side B is pickier and weights ambition more.

| Variable | Hidden? | How it works | Placeholder defaults |
|---|---|---|---|
| General appeal `u` | Hidden | How attractive someone is to people in general (mostly looks). Users react to it; models never see it | Standard normal per person |
| Looks-vs-type weight `w` | Hidden | How much this person cares about general appeal vs. their own type. Per person, drawn around a side average | A ≈ 0.6, B ≈ 0.5, spread ±0.15, clipped to [0, 1] |
| Traits (5, below) | **Displayed** | What profiles show, and what models are allowed to use | Values in [0, 1] |
| Ideal per ideal-point trait | Hidden | Partly resembles own traits: `ideal = ρ·own + (1 − ρ)·random` | ρ = 0.5 |
| Trait importance | Hidden | Per person, around side-level averages (table below) | ±30% individual variation |
| Like-rate (pickiness) | Hidden | Target fraction of profiles this person likes. Drawn from a side distribution, then shifted lower for higher `u` (attractive people are pickier) | A: Beta(2, 3), mean 0.40, wide spread. B: Beta(4, 36), mean 0.10, narrow. Shift ≈ −0.05 per SD of `u`, clipped to [0.02, 0.9] |
| Attention budget | Hidden | Incoming likes this person reviews per day | 5–15 per person |
| Pair chemistry | Hidden | Fixed random term per pair (one-off chemistry). Not re-drawn per swipe | Normal, small SD |

**Pickiness vs. specificity:** like-rate is *how many* people you like. `w` plus trait importance is *how narrow your type is*. They're separate on purpose: you can be very specific about your type without having a low like-rate, and vice versa.

### The traits

| Trait | Type | Side A avg importance | Side B avg importance | Notes |
|---|---|---|---|---|
| Relationship intent (casual ↔ serious) | Closer to ideal | Moderate | Moderate | Real Hinge profile field |
| Travel / adventurousness (settled ↔ always moving) | Closer to ideal | Moderate | Moderate | |
| Religiosity (not religious ↔ very) | Closer to ideal | Moderate | Moderate | Chosen over politics: the speed-dating data has a same-religion importance field for calibration |
| Fitness | More is better | High | High, slightly lower | Correlated ~0.4 with `u`: a displayed trait that partly reveals the hidden one (like photo features on real apps) |
| Ambition / career | More is better | Moderate | Moderate, somewhat higher | |

Watch: fitness is weighted heavily by both sides *and* correlated with `u`, so it behaves partly like a second universal score. If charts show extreme concentration even at low `w`, check the fitness weights first.

### True preference matrices (the heart of the project)
```python
# taste: how well B fits A's type
taste(A, B) = Σ_ideal-point traits  imp_A,k · (1 − |ideal_A,k − trait_B,k|)
            + Σ_more-is-better traits imp_A,k · trait_B,k

# standardize both parts so w means what it says
score(A→B) = w_A · z(u_B) + (1 − w_A) · z_row(taste(A, ·))[B] + chemistry(A, B)

# threshold from like-rate: A likes their top like_rate_A fraction
threshold_A = quantile(score(A→·), 1 − like_rate_A)
P[a, b] = sigmoid((score − threshold_a) / temp)      # A likes B
Q[b, a] = same construction, other direction          # B likes A back
```
- Compute `P` and `Q` once, fully vectorized in NumPy. Never loop over pairs in Python.
- `temp` ≈ 0.3 in standardized units (placeholder). Check realized like-rates match targets.
- **The `w` sweep:** shift both sides' average `w` together from low to high (keeping the side gap), e.g. side-A average from 0.1 to 0.9. This is the x-axis of the main charts (same role as λ in the MODE paper). For demos, side-A average ≈ 0.6–0.7.

### Phase 1a — Classroom mode (daily loop)
Days run in **synchronized rounds**: everyone reviews their inbox, then everyone browses. Likes sent today arrive in the recipient's inbox tomorrow.

1. **Inbox review:** each user reviews incoming likes **best-first** (by their own `P`/`Q`), up to their attention budget, and likes back with true probability. Like-back = match. **Liking back is free** (doesn't use the daily like cap). **Unreviewed likes vanish** at the end of the day.
2. **Browsing:** the ranker returns a list of **10 profiles**, excluding anyone already seen, already matched, or currently in the user's inbox. The user looks at all 10 and likes each with true probability.
3. **Daily like cap (8):** rarely binding in classroom mode (10 profiles × like-rate); it matters in Hinge mode.
4. **Log every event:** `day, viewer, shown, position, liked, matched, ranker`, plus inbox events (`received, reviewed, liked_back`).

Matched pairs never see each other again; all users stay in the market for the whole run.

This log is the training data for Phases 3–4.

### Phase 1b — Hinge mode (add one at a time)
Each is a config switch. After adding each one, re-run the baseline sweep and record what changed.

1. **Scrolling:** ranker returns an ordered list of up to 50; the user scrolls until they've used their **8 daily likes** or lose patience (**~5% chance of quitting after each profile**, ~20 profiles on average). Expected emergent effect: side A hits the like cap; side B runs out of patience first and sends far fewer likes. Nobody codes that in.
2. **Newest-first inbox:** incoming likes are reviewed newest first instead of best-first.
3. **Carry-over forever:** unreviewed likes stay in the inbox. Combined with newest-first, popular users' old likes are effectively dead: timing matters as much as compatibility.

Exposure check for scrolling: ~20 profiles/day × 30 days at 2,000/side ≈ 30% of the pool.

### Rankers
- **Random** — the floor.
- **Popularity** — most likes received first; random tie-breaking (day 1 everyone has zero).
- **Elo** — when A swipes on B, **only B's rating changes** (like = win, pass = loss, weighted by A's rating). Start at 1500, standard update. Show candidates with a similar **percentile within their own side**, not similar raw rating: the sides' ratings aren't comparable because side B is pickier.
- **One-sided oracle** — rank by `P[a, b]` only (how much A would like B, ignoring whether B likes A back).
- **Reciprocal oracle** — rank by `P[a, b] · Q[b, a]` using the true matrices. The best *pairwise* ranker, **not** the best possible: it ignores congestion, so MODE may beat it in Phase 2.
- **Exploration:** popularity and Elo get ~10% random slots, so users who start low aren't starved forever.
- *Stretch:* **Gale-Shapley** (Hinge's old "Most Compatible"). Stable matching: one side proposes in order of preference, the other holds its best offer and rejects the rest, repeat until stable. ~30 lines. Returns one partner per person, not a list, so evaluate it as a daily top pick only.

### Metrics
- Total matches
- Gini of matches per user; Gini of likes received
- % of users with zero matches
- Matches by `u` decile (the "who gets left behind" chart)
- **Likes never reviewed** (share of sent likes that died in an inbox) — the clearest single congestion measure
- Realized like-rate per side; profiles viewed per user, by side (Hinge mode)

### Sanity checks before trusting anything
- High `w` → likes concentrate heavily at the top. Low `w` → much flatter. If not, it's a bug.
- Reciprocal oracle ≥ random, popularity, and Elo on total matches.
- Reciprocal oracle > one-sided oracle on total matches.
- Realized like-rates per side ≈ the targets.
- Symmetric config (both sides' numbers equal) → the two sides' metrics come out roughly equal.

### Experiments and deliverables
- `experiments/01_baselines_sweep.py` (classroom mode): every ranker × `w` sweep × seeds →
  - matches vs. `w`, one line per ranker
  - Gini of matches vs. `w`
  - matches by `u` decile at a high `w`
  - one-sided vs. reciprocal oracle (why reciprocal recommendation exists)
- `experiments/01b_hinge_mode.py`: one chart per added mechanic, showing what it changed (especially dead likes and profiles viewed by side)
- `experiments/01c_scale_check.py`: key results at 500 / 1,000 / 2,000 per side, to confirm they don't flip with market size
- README section: setup, charts, three findings, limitations

---

## Phase 2 — MODE (market-level ranking)

**Paper:** Tomita, *MODE: Mutual Optimality in Direct Effects of Reciprocal Recommendations in Matching Markets*, RecSys '26. arXiv:2608.01731. Code: github.com/CyberAgentAILab/mode
**Read:** §2 (setup + older methods), §3 (MODE), §4.1 (synthetic experiment), Appendix B (pseudocode for TU / ApproxSW / DirectSW).
**Reference only if §2 doesn't click:** Su, Bayoumi & Joachims (2022), *Optimizing Rankings for Recommendation in Matching Markets*, arXiv:2106.01941.

### Concepts to understand by the end
- **Proactive vs. reactive side:** one side sends likes from recommendations; the other reviews incoming likes in order of their own preference.
- **Position-based examination:** the chance someone even looks at slot k (e.g. 1/k). Explained in MODE §2.1.
- **Social welfare** = expected total matches across the platform.
- **Direct vs. indirect effects:** your own list (visible to you) vs. how crowded your targets' inboxes are (invisible to you).
- **MODE's idea:** repeatedly give each user the list that maximizes *their own* expected matches, given everyone else's current lists, until it stabilizes (a Nash-equilibrium-like fixed point).

### Step A — Full standalone reproduction of Figure 1a (no simulator)
- Implement their synthetic generator exactly: `p_ij = (1 − λ)·U[0,1] + λ·popularity_j`, candidates = 1.5n, n = 50, examination types log / inv / exp, λ from 0 to 1.
- Implement from pseudocode: Naive, Reciprocal (product), TU, ApproxSW, DirectSW, MODE. Keep the two stochastic methods (ApproxSW, DirectSW) at n ≤ 50; the paper reports >10 hours at n = 100.
- Implement their exact expected-matches computation (Algorithm 1, ranking probabilities).
- **Target:** reproduce Figure 1a. At high λ, MODE should roughly match the slow stochastic methods and beat Naive / Reciprocal / TU.
- Only look at their repo *after* my version runs, to check discrepancies. Document any gaps and why.

### Step B — MODE inside my simulator
- Feed the simulator's true `P` and `Q` into MODE (no learning yet).
- **Classroom mode first:** it already matches MODE's assumptions (fixed lists, best-first inbox review, nothing carries over). Remaining mismatch: both sides browse in my simulator; MODE assumes one proactive side. Start with side A proactive to match the paper, then try both.
- **Then Hinge mode:** scrolling, newest-first review, and carry-over all break MODE's assumptions. Does its advantage survive?
- Decide how often to recompute MODE (daily is fine at 2,000×2,000 with 10-item lists; vectorize the ranking-probability loop).
- Compare against Phase 1 rankers (including the reciprocal oracle) on matches, Gini, zero-match %, dead likes, per-user regret (MODE's sub-optimality metric), and runtime.

### Deliverables
- `experiments/02_mode_repro.py` + my Figure 1a next to theirs
- `experiments/03_mode_in_sim.py` + comparison table (classroom and Hinge mode)
- Paper note (template below)

---

## Phase 3 — Matrix factorization (estimating preferences from logs)

**Learn from:** any ALS / implicit-feedback MF tutorial (e.g. the `implicit` library docs). No paper needed.

- Train one MF model per direction on the Phase 1 logs: A→B likes, B→A likes. This is what Hinge ran for years, and what MODE's author used on real data.
- Output estimated `P̂` and `Q̂`.
- **Evaluate estimation:** correlation and AUC of `P̂` vs. true `P` on held-out pairs.
- **Evaluate downstream:** run Naive / Reciprocal / MODE using `P̂, Q̂` instead of the oracle. How many matches does estimation error cost?

### Exposure bias
Logs only contain pairs that some ranker chose to show. Train once on logs from the *random* ranker and once on logs from *Elo*, and compare. Expect the Elo-trained model to be confidently wrong about people Elo never showed. Don't evaluate a model only on data from the policy that generated its training logs.

### Deliverables
- `experiments/04_mf.py`, `experiments/06_exposure_bias.py`
- README section with estimation and downstream results

---

## Phase 4 — Reciprocal two-tower (Hinge-style)

**Paper:** *Multi-Objective Candidate Generation for Reciprocal Recommendation at Hinge* (2026), ACM, doi:10.1145/3774935.3807906
**Optional reference for training tricks:** Yi et al. (2019), *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (in-batch negatives, popularity correction).

### Architecture (from the Hinge paper)
- **Rater tower** and **profile tower**, each with a user-ID embedding plus displayed-trait features.
- Two predictions per pair: rater likes profile; profile likes rater back.
- Match probability = product of the two. Trained multi-task.

### Implementation details
- Features: user ID + the 5 displayed traits only. **Never** feed `u`, `w`, ideals, importance, like-rate, or attention budget; that's leakage. The model must learn them from behavior through the ID embeddings.
- Check whether the model learns to use fitness as a partial proxy for hidden `u`.
- Labels: shown-and-liked = 1, shown-and-passed = 0; add in-batch random negatives.
- Train on the M-series Mac with PyTorch MPS; 2,000×2,000 is small.
- **Cold start test:** hold out "new users" with traits but no history. Use ID-embedding dropout in training so the model can fall back on traits. (Hinge mentions real-time embeddings for first-session users; this is the toy version.)

### Evaluation
- Estimation: AUC / correlation vs. true `P` and `Q`, MF vs. two-tower, warm vs. cold users.
- Downstream: two-tower scores → MODE, compared with MF → MODE and oracle → MODE.
- **Key result:** the oracle-vs-learned gap. How much does imperfect estimation cost a market-level ranker?
- Offline vs. online: does the model with the best offline AUC produce the most matches when deployed in the simulator? (It may not.)

### Deliverables
- `experiments/05_two_tower.py`
- README section with estimation, cold-start, and downstream results

---

## Repo structure
```
dating-sim/
  sim/
    config.py          # every number, per side
    population.py      # generate users + traits
    preferences.py     # true P, Q
    market.py          # daily loop (classroom + Hinge-mode switches) + logging
    metrics.py
  rankers/
    baselines.py       # random, popularity, one-sided oracle, reciprocal oracle, naive, reciprocal
    elo.py
    gale_shapley.py
    tu.py
    approx_sw.py       # Phase 2 reproduction only (small n)
    mode.py
  models/
    mf.py
    two_tower.py
  experiments/
    01_baselines_sweep.py
    01b_hinge_mode.py
    01c_scale_check.py
    02_mode_repro.py
    03_mode_in_sim.py
    04_mf.py
    05_two_tower.py
    06_exposure_bias.py
  notes/papers/        # one note per paper
  results/             # charts + CSVs, regenerated by experiments
  README.md
```

## Paper-note template (one per paper, in `notes/papers/`)
```
Problem:          what it solves, one sentence
Method:           how, one sentence
Assumes:          what it takes for granted
Didn't test:      what it leaves out
In my simulator:  what matches, what doesn't, what I'll try
```

---

## Backlog (after Phase 4)
- **Calibration against real data:**
  - **Libimseti** (Czech dating site, ~17M user-to-user ratings): tune `w` until the simulated likes-received distribution roughly matches.
  - **Columbia speed-dating data** (Fisman et al., ~8k dates, both people's decisions + rated traits, same-religion importance): tune side like-rates and trait importance.
  - Optional: run the two-tower directly on Libimseti to confirm it trains on real data.
- **Dynamic pickiness:** people get pickier as they receive more likes, so rankers that flood popular users make them pickier (a feedback loop). Breaks the fixed ground truth, so only after Phases 2–4.
- **Activity level:** daily login probability; inactive users' inboxes rot.
- **LLM onboarding interview** → text embedding as a two-tower feature; measure cold-start lift. Toy version of Tinder Chemistry / Bumble Bee.
- **Cold start / travel mode:** landing in a new city as a fresh profile; new-user boosts; how fast a new profile's estimates settle.
- **Bandits** for exploring new users (Thompson sampling, epsilon-greedy).
- **Fairness:** Tomita & Yokoyama (2024), *Fair Reciprocal Recommendation in Matching Markets*.
- **Photo quality:** how much better photos move outcomes under each ranker.
