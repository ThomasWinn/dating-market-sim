# Dating Market Simulator — Project Notes
*Brainstormed Sep 26, 2026. Ideas, not rules.*

## Why
- Answer a question I've always had: **how do dating apps decide who sees whom, and why do people get ghosted?**

## The big idea
A synthetic, Hinge-style dating market. Two connected parts:

1. **Ranking:** who gets shown to whom (Elo, stable matching, learned models).
2. **Conversations:** after a match, what happens? Who ghosts, and why?

Part 1 feeds Part 2: a lot of ghosting is market-driven (too many matches, not enough attention), not just personality clash.

## Profiles (no images)
**Hidden scores:** universal attractiveness, photo quality (separate and improvable), pickiness, activity level, attention budget (how many chats someone can keep up).

**Displayed traits (0–1, keep it to ~8):** goofy/playful, aloof/"stuck-up" look, flex/status display, wealth/career signals, warmth, fitness, adventurous/traveler, nightlife vs. homebody.

**Preferences:** each user has weights over those traits. Each trait has an *average appeal* and a *polarization* (how much people disagree).

**Attraction:**
```
attraction(A→B) = w · universal(B) + (1 − w) · taste_match(A, B) + noise
```
`w` is a knob. Sweeping it is one of the best experiments.

**For Part 2:** interests/hobbies, intent (long-term / casual / unsure), dealbreakers (age, distance) as hard filters. LLM writes each profile's prompt answers *from* its traits.

## Part 1: Ranking
- Algorithms to compare: random, Elo, stable-matching-style (Hinge "Most Compatible"), a learned **two-tower model in PyTorch** predicting *mutual* likes.
- Hinge mechanics worth modeling: likes aimed at a specific photo/prompt, daily like caps, Roses, Standouts.
- Metrics: match rate, inequality (Gini), matches that never message, churn.
- Watch for offline vs. online gap: a model that looks good on logs can make the market worse.

## Part 2: Conversations & ghosting
- Each agent knows its competing matches and attention budget.
- Ghosting should *emerge* (better option arrives, inbox overload, interest decay), not be scripted.
- Hypotheses to test: Does Elo cause more ghosting at the top? Do exposure caps help? Do people matching "up" get ghosted more?
- Optional: LLM agents with personas actually chatting (run a small local model on the M4). Caveat: this shows the LLM's idea of dating, not real humans. Framing: "do AI agents reproduce known human patterns?"

## Spin-offs (same codebase)
- **Travel mode:** landing in a new city as a fresh profile. New-user boost, how fast your score settles.
- **Bandits:** exploring new users' desirability (Thompson sampling, epsilon-greedy).
- **Gaming the system:** right-swipe-everyone users. Which algorithms are robust?
- **Photo quality:** how much can better photos move your ranking under each algorithm?

## Ground rules
- No real people's messages or photos.
- No images, no face-attractiveness model.
- One repo, modular. Depth over breadth.
- Write up findings with real charts and numbers.

## Possible first weekend
Population generator → random vs. Elo ranking → plot match rate and inequality. No neural nets yet.

## Open questions
- What should `w` default to?
- How realistic does the population need to be?
- Where do LLM agents actually earn their cost?
- What result would be most surprising?
