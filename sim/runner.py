"""Run many simulations in parallel and collect one metrics row per (config, seed, ranker)."""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from rankers.baselines import (
    OneSidedOracle,
    PopularityRanker,
    RandomRanker,
    ReciprocalOracle,
)
from rankers.elo import EloRanker
from rankers.gale_shapley import GaleShapleyRanker
from sim.config import Config
from sim.market import simulate
from sim.metrics import summarize
from sim.preferences import build_world

BASELINES = [
    r.name for r in (RandomRanker, PopularityRanker, EloRanker, OneSidedOracle, ReciprocalOracle)
]
RANKERS = {
    r.name: r
    for r in (
        RandomRanker,
        PopularityRanker,
        EloRanker,
        OneSidedOracle,
        ReciprocalOracle,
        GaleShapleyRanker,
    )
}


def _run(task: tuple[dict, Config, int, list[str]]) -> list[dict]:
    tags, cfg, seed, rankers = task
    world = build_world(cfg, seed)  # built once, shared by every ranker (same people, same coins)
    rows = []
    for name in rankers:
        start = time.perf_counter()
        res = simulate(world, RANKERS[name]())
        seconds = time.perf_counter() - start
        rows.append(
            {**tags, "seed": seed, "ranker": name, "seconds": seconds, **summarize(world, res)}
        )
    return rows


def run_grid(
    jobs: list[tuple[dict, Config]],
    seeds: range,
    rankers: list[str] = BASELINES,
    workers: int | None = None,
) -> pd.DataFrame:
    """`jobs` is a list of (tags, config); tags (e.g. {"w": 0.3}) are copied onto each row."""
    tasks = [(tags, cfg, seed, rankers) for tags, cfg in jobs for seed in seeds]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers) as pool:
        rows = [row for chunk in pool.map(_run, tasks) for row in chunk]
    return pd.DataFrame(rows)
