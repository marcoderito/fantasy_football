# -*- coding: utf-8 -*-
"""hyperparameter_tuning.py – **robust version**

This edition fixes the two issues you just hit:

1. **TypeError creator.create()** – we now create the DEAP classes once at
   module load (and only if they do not yet exist).
2. **“No bids were made”** – the auction logic needs at least a *second*
   manager to generate competitive bids.  We add a lightweight dummy
   opponent with a built‑in *random* strategy.  Your test manager is still
   the one we measure.

Run with:
    python hyperparameter_tuning.py

Outputs (in *results/*):
    – hyperparam_results.csv  (table)
    – hyperparam_plot.png     (bar chart)
"""

from __future__ import annotations

import copy
import itertools
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from deap import algorithms, base, creator, tools
from unittest.mock import patch

# ---------------------------------------------------------------------------
# 0.  GLOBAL SETTINGS
# ---------------------------------------------------------------------------

DATA_FILE = "Fantacalcio_stat.csv"       # <- must exist in repo
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
GLOBAL_SEED = 42                          # reproducibility

# Role constraints – identical min & max as requested
ROLE_CONSTRAINTS = {"P": (3, 3), "D": (8, 8), "C": (8, 8), "A": (6, 6)}
BUDGET = 500
MAX_TOTAL = sum(mx for _, mx in ROLE_CONSTRAINTS.values())  # 25 players

# ---------------------------------------------------------------------------
# 1.  LOAD PLAYERS ONCE
# ---------------------------------------------------------------------------

from data_loader import load_data
from utils import Player, Manager, score_player
import optimization as opt
from optimization import (
    multi_manager_auction,
    max_bid_possible,
    max_bid_for_player,
    min_bid_threshold,
    common_fitness_logic,
)

players_df = load_data(DATA_FILE)
PLAYERS_MASTER: List[Player] = [
    Player(
        pid,
        row["Name"], row["Role"],
        goals_scored=row["Goals_Scored"],
        assists=row["Assists"],
        yellow_cards=row["Yellow_Cards"],
        red_cards=row["Red_Cards"],
        rating=row["Rating"],
    )
    for pid, row in players_df.iterrows()
]

# ---------------------------------------------------------------------------
# 2.  DEAP CREATOR CLASSES  (defined only once)
# ---------------------------------------------------------------------------

if "FitnessMinTune" not in creator.__dict__:
    creator.create("FitnessMinTune", base.Fitness, weights=(-1.0,))
if "IndividualTune" not in creator.__dict__:
    creator.create("IndividualTune", list, fitness=creator.FitnessMinTune)

# ---------------------------------------------------------------------------
# 3.  HYPER‑PARAMETER GRIDS  (values from lecture + extras)
# ---------------------------------------------------------------------------

DE_POPSIZE   = [10, 15, 20]
DE_MUT       = [(0.5, 1.0), (0.7, 1.2)]
DE_RECOMB    = [0.7, 0.9]

PSO_PART     = [30, 60]
PSO_W        = [0.9, 0.5]
PSO_C1       = [1.49445]
PSO_C2       = [1.49445]

ES_MU        = [15, 20]
ES_LAMBDA    = [30, 40]
ES_NGEN      = [50, 80]

# ---------------------------------------------------------------------------
# 4.  CUSTOM ES STRATEGY  (μ, λ, ngen exposed)
# ---------------------------------------------------------------------------

def es_generate_bids(
    manager: Manager,
    unassigned: List[Player],
    mu: int,
    lambda_: int,
    ngen: int,
):
    """Return a list (pid, bid) using (μ, λ)-ES with given params."""

    if not unassigned or manager.budget <= 0:
        return []

    max_each = min(max_bid_possible(manager), max_bid_for_player(manager))
    if max_each < 1:
        return []

    roles   = [p.role for p in unassigned]
    scores  = [score_player(p) for p in unassigned]
    pids    = [p.pid  for p in unassigned]
    min_thr = min_bid_threshold(manager)

    toolbox = base.Toolbox()
    toolbox.register("attr_bid", random.uniform, 0, float(max_each))
    toolbox.register("individual", tools.initRepeat, creator.IndividualTune,
                     toolbox.attr_bid, len(unassigned))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_ind(ind):
        return (common_fitness_logic(manager, ind, roles, scores, min_thr),)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=0, sigma=max_each/6, indpb=0.2)
    toolbox.register("select", tools.selBest)

    pop = toolbox.population(mu + lambda_)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=mu, lambda_=lambda_,
                              cxpb=0.5, mutpb=0.3, ngen=ngen, verbose=False)
    best = tools.selBest(pop, 1)[0]
    bids = [max(min_thr, int(round(b))) if 0 < b < min_thr else int(round(b))
            for b in best]
    return [(pids[i], b) for i, b in enumerate(bids) if 0 < b <= manager.budget]

# ---------------------------------------------------------------------------
# 5.  SINGLE EXPERIMENT
# ---------------------------------------------------------------------------

def run_experiment(algo: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any], float]:
    """Run one auction with the chosen algorithm and parameters."""

    players = copy.deepcopy(PLAYERS_MASTER)

    # Test manager (the one we measure)
    test_mgr = Manager("TEST", BUDGET, ROLE_CONSTRAINTS, MAX_TOTAL, algo)

    # Dummy rival manager with simple random strategy – ensures bids exist
    rival   = Manager("RIVAL", BUDGET, ROLE_CONSTRAINTS, MAX_TOTAL, "random")

    patch_ctx = None  # default

    if algo == "pso":
        original_pso = opt.pso
        def patched(func, lb, ub, **k):
            return original_pso(func, lb, ub,
                                swarmsize=params["particles"],
                                maxiter=80,
                                omega=params["w"],
                                phip=params["c1"], phig=params["c2"])
        patch_ctx = patch.object(opt, "pso", new=patched)

    elif algo == "de":
        original_de = opt.differential_evolution
        def patched(func, bounds, **k):
            return original_de(func, bounds,
                               strategy="best1bin", maxiter=80, popsize=params["popsize"],
                               mutation=params["mutation"],
                               recombination=params["recomb"])
        patch_ctx = patch.object(opt, "differential_evolution", new=patched)

    elif algo == "es":
        def decide_bids(unassigned):
            return es_generate_bids(test_mgr, unassigned,
                                    mu=params["mu"], lambda_=params["lambda"], ngen=params["ngen"])
        test_mgr.decide_bids = decide_bids  # type: ignore[assignment]

    # ---- run auction ----
    random.seed(GLOBAL_SEED)
    if patch_ctx:
        with patch_ctx:
            managers_after, _ = multi_manager_auction(players, [test_mgr, rival], max_turns=60)
    else:
        managers_after, _ = multi_manager_auction(players, [test_mgr, rival], max_turns=60)

    fit = sum(score_player(p) for p in managers_after[0].team)
    return algo.upper(), params, fit

# ---------------------------------------------------------------------------
# 6.  GRID SEARCH
# ---------------------------------------------------------------------------

results: List[Tuple[str, Dict[str, Any], float]] = []

# DE
for pop, mut, rec in itertools.product(DE_POPSIZE, DE_MUT, DE_RECOMB):
    results.append(run_experiment("de", {"popsize": pop, "mutation": mut, "recomb": rec}))

# PSO
for part, w, c1, c2 in itertools.product(PSO_PART, PSO_W, PSO_C1, PSO_C2):
    results.append(run_experiment("pso", {"particles": part, "w": w, "c1": c1, "c2": c2}))

# ES
for mu, lam, ngen in itertools.product(ES_MU, ES_LAMBDA, ES_NGEN):
    results.append(run_experiment("es", {"mu": mu, "lambda": lam, "ngen": ngen}))

# ---------------------------------------------------------------------------
# 7.  SAVE & DISPLAY
# ---------------------------------------------------------------------------

results.sort(key=lambda r: r[2], reverse=True)

df = pd.DataFrame([{**{"algo": a}, **p, "fitness": f} for a, p, f in results])
csv_path = RESULTS_DIR / "hyperparam_results.csv"
df.to_csv(csv_path, index=False)
print("\nSaved", csv_path)
print("\n=== TOP 10 CONFIGURATIONS ===")
print(df.head(10).to_string(index=False))

# Simple bar chart
plt.figure(figsize=(10, 5))
plt.bar(range(len(results)), [f for _, _, f in results])
plt.xticks(range(len(results)), [f"{a}-{i}" for i, (a, _, _) in enumerate(results, 1)],
           rotation=90, fontsize=6)
plt.ylabel("Team fitness")
plt.tight_layout()
plot_path = RESULTS_DIR / "hyperparam_plot.png"
plt.savefig(plot_path, dpi=120)
print("Plot saved to", plot_path)
