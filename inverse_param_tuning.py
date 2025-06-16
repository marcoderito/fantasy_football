# -*- coding: utf-8 -*-
"""
inverse_multi_tuning.py  (2025-06-15)

Outer-level PSO that select:
    • which internal meta-heuristics to use  (PSO / DE / ES)
    • its main hyperparameters
to match scores, forced-picks, and leftovers with a target profile.

Outputs:
    inverse_trace.csv
    plot/inv_loss_curve.png
    plot/inv_compare.png
    plot/inv_summary.csv
"""

from __future__ import annotations
import copy, csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarm import pso                                # outer optimiser
from unittest.mock import patch
from deap import base, creator, tools, algorithms

# ------------------------------------------------------------------ #
# Domain-specific imports
# ------------------------------------------------------------------ #
from data_loader import load_data
from utils import Player, Manager, score_player
import optimization as opt
from optimization import (
    multi_manager_auction,
    max_bid_possible, max_bid_for_player,
    min_bid_threshold, common_fitness_logic
)

# ------------------------------------------------------------------ #
# 0. GLOBAL CONFIG
# ------------------------------------------------------------------ #
DATA_FILE = "Fantacalcio_stat.csv"
GLOBAL_SEED = 42

ROLE_CONSTRAINTS = {"P": (3, 3), "D": (8, 8), "C": (8, 8), "A": (6, 6)}
BUDGET, MAX_TOTAL = 500, 25

TARGET_SCORE, TARGET_FORCED, TARGET_LEFT = 100, 4, 0
TARGET_VEC = np.array([TARGET_SCORE, TARGET_FORCED, TARGET_LEFT])

TRACE = Path("inverse_trace.csv")
if not TRACE.exists():
    with open(TRACE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["alg","p1","p2","p3","p4",
             "score","forced","leftover"]
        )

PLOT_DIR = Path("plot"); PLOT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------ #
# 1. LOAD PLAYERS ONCE
# ------------------------------------------------------------------ #
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

# ------------------------------------------------------------------ #
# 2. DEAP classes (for ES) only once
# ------------------------------------------------------------------ #
if "FitnessMinInv" not in creator.__dict__:
    creator.create("FitnessMinInv", base.Fitness, weights=(-1.0,))
if "IndividualInv" not in creator.__dict__:
    creator.create("IndividualInv", list, fitness=creator.FitnessMinInv)

# ------------------------------------------------------------------ #
# 3. Custom ES generator
# ------------------------------------------------------------------ #
def es_generate_bids(manager: Manager,
                     unassigned: List[Player],
                     mu: int, lam: int, ngen: int):
    if not unassigned or manager.budget <= 0:
        return []
    max_each = min(max_bid_possible(manager), max_bid_for_player(manager))
    if max_each < 1:
        return []

    roles   = [p.role for p in unassigned]
    scores  = [score_player(p) for p in unassigned]
    pids    = [p.pid  for p in unassigned]
    min_thr = min_bid_threshold(manager)

    tb = base.Toolbox()
    tb.register("attr_bid", np.random.uniform, 0, float(max_each))
    tb.register("individual", tools.initRepeat,
                creator.IndividualInv, tb.attr_bid, len(unassigned))
    tb.register("population", tools.initRepeat, list, tb.individual)

    def eva(ind):
        return (common_fitness_logic(manager, ind, roles, scores, min_thr),)
    tb.register("evaluate", eva)
    tb.register("mate", tools.cxBlend, alpha=0.3)
    tb.register("mutate", tools.mutGaussian,
                mu=0, sigma=max_each/6, indpb=0.2)
    tb.register("select", tools.selBest)

    pop = tb.population(mu + lam)
    algorithms.eaMuPlusLambda(pop, tb,
                              mu=mu, lambda_=lam,
                              cxpb=0.5, mutpb=0.3,
                              ngen=ngen, verbose=False)
    best = tools.selBest(pop, 1)[0]
    bids = [max(min_thr, int(round(b))) if 0 < b < min_thr else int(round(b))
            for b in best]
    return [(pids[i], b) for i, b in enumerate(bids)
            if 0 < b <= manager.budget]

# ------------------------------------------------------------------ #
# 4. Inner evaluation
# ------------------------------------------------------------------ #
def eval_inner(algo_id: int, p1: float, p2: float, p3: float, p4: float
               ) -> Tuple[float,int,int]:
    """
    Return (fitness, forced_picks, leftover) per:
        algo_id = 0 -> PSO   (p1=ω, p2=c1, p3=c2, p4=swarm)
                 1 -> DE    (p1=pop, p2=F_low, p3=F_high, p4=CR)
                 2 -> ES    (p1=μ,  p2=λ,    p3=ngen)
    """
    algo_id = int(round(algo_id)) % 3
    players = copy.deepcopy(PLAYERS_MASTER)

    test  = Manager("TEST",  BUDGET, ROLE_CONSTRAINTS, MAX_TOTAL, "custom")
    rival = Manager("RIVAL", BUDGET, ROLE_CONSTRAINTS, MAX_TOTAL, "random")

    # ----- PSO ------------------------------------------------------
    if algo_id == 0:
        omega, c1, c2 = p1, p2, p3
        swarm = int(round(p4))
        orig_pso = opt.pso
        def patched(func, lb, ub, **k):
            return orig_pso(func, lb, ub,
                            swarmsize=swarm, maxiter=80,
                            omega=omega, phip=c1, phig=c2)
        with patch.object(opt, "pso", new=patched):
            multi_manager_auction(players, [test, rival], max_turns=60)

    # ----- DE -------------------------------------------------------
    elif algo_id == 1:
        pop     = int(round(p1))
        F_low   = max(0.1, min(p2, p3))
        F_high  = max(F_low + 0.1, max(p2, p3))
        CR      = min(1.0, max(0.1, p4))
        orig_de = opt.differential_evolution
        def patched(func, bounds, **k):
            return orig_de(func, bounds,
                           strategy="best1bin", maxiter=80, popsize=pop,
                           mutation=(F_low, F_high), recombination=CR)
        with patch.object(opt, "differential_evolution", new=patched):
            multi_manager_auction(players, [test, rival], max_turns=60)

    # ----- ES -------------------------------------------------------
    else:
        mu   = int(round(max(4, p1)))
        lam  = int(round(max(mu+1, p2)))
        ngen = int(round(max(10, p3)))
        def decide(unassigned):
            return es_generate_bids(test, unassigned, mu, lam, ngen)
        test.decide_bids = decide               # type: ignore
        multi_manager_auction(players, [test, rival], max_turns=60)

    fitness  = sum(score_player(p) for p in test.team)
    forced   = sum(1 for p in test.team
                   if getattr(p, "final_price", 0) == 1)
    leftover = test.budget
    return fitness, forced, leftover

# ------------------------------------------------------------------ #
# 5. LOSS FUNCTION per outer PSO
# ------------------------------------------------------------------ #
def loss(theta: np.ndarray) -> float:
    alg, p1, p2, p3, p4 = theta
    score, forced, left = eval_inner(alg, p1, p2, p3, p4)

    # log
    with open(TRACE, "a", newline="") as f:
        csv.writer(f).writerow(
            [int(round(alg)), p1, p2, p3, p4, score, forced, left]
        )

    return abs(score  - TARGET_SCORE) \
         + abs(forced - TARGET_FORCED) \
         + abs(left   - TARGET_LEFT)

# ------------------------------------------------------------------ #
# 6. OUTER-LEVEL PSO SEARCH
# ------------------------------------------------------------------ #
LB = [-0.49,   0.3,  0.3,  0.3,  10]    # alg, p1..p4
UB = [ 2.49,   2.5,  2.5,  2.5, 100]

best_theta, best_loss = pso(
    loss, LB, UB,
    swarmsize=30, maxiter=40,
    omega=0.7, phip=1.5, phig=1.5
)

print("\n=== Inverse multi-tuning result ===")
print("Best theta:", best_theta)
print("Loss:", best_loss)

# ------------------------------------------------------------------ #
# 7. POST-PROCESSING
# ------------------------------------------------------------------ #
df = pd.read_csv(TRACE)

# calculates the L1-loss at each iteration
loss_curve = np.abs(df[["score","forced","leftover"]].values - TARGET_VEC).sum(axis=1)


plt.figure(); plt.plot(loss_curve)
plt.xlabel("Iteration"); plt.ylabel("L1 loss"); plt.tight_layout()
plt.savefig(PLOT_DIR / "inv_loss_curve.png", dpi=120)

before = df.iloc[0]; after = df.iloc[-1]
summary = pd.DataFrame({
    "metric":["score","forced","leftover"],
    "before":[before["score"], before["forced"], before["leftover"]],
    "after" :[after ["score"],  after ["forced"],  after ["leftover"]],
    "target":[TARGET_SCORE, TARGET_FORCED, TARGET_LEFT]
})
summary.to_csv(PLOT_DIR / "inv_summary.csv", index=False)

plt.figure()
x = np.arange(3)
plt.bar(x-0.2, summary["before"], width=0.4, label="before")
plt.bar(x+0.2, summary["after"],  width=0.4, label="after")
plt.xticks(x, summary["metric"]); plt.legend(); plt.tight_layout()
plt.savefig(PLOT_DIR / "inv_compare.png", dpi=120)

print("Trace, plots & summary saved in", PLOT_DIR)
