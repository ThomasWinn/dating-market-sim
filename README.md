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

- [ ] Random vs. Elo ranking: match rate and inequality (Gini)
- [ ] Stable-matching-style "most compatible" picks
- [ ] A learned ranker that predicts *mutual* likes
- [ ] The knob `w`: how much attraction is universal vs. individual taste
- [ ] Ghosting that emerges on its own from attention budgets and competing matches
- [ ] Side quests: new-user boosts, swipe-right-on-everyone users, how much better photos move you

## Ground rules

- Synthetic people only: no real profiles, messages, or photos.
- No images and no face-attractiveness model. Everything is numbers and traits.

## Status

Early days. Nothing runs yet.

## Dev

```bash
uv sync
uv run pytest
```
