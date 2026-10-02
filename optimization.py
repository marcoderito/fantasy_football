"""
Auction Optimisation Engine

Provides the entire AI core used by the auction system:
  • Shared fitness function with budget, role, and surrogate-model logic
  • Three meta-heuristics to build bid vectors
        1. Particle Swarm Optimisation (pyswarm)
        2. Differential Evolution (SciPy)
        3. (μ+λ) Evolution Strategy (DEAP)
  • Conflict resolution helper (mini-auction with dynamic re-bids)
  • Full multi-manager auction loop with forced assignments

All functions are stateless; Managers and Players carry their own data.

Author: Marco De Rito
"""

import numpy as np
import random
from typing import List, Tuple
from utils import score_player
from deap import base, creator, tools, algorithms
from pyswarm import pso
from scipy.optimize import differential_evolution


class FitnessMax(base.Fitness):
    """Fitness class for minimization problems."""
    weights = (-1.0,)


class Individual(list):
    """Individual which holds a list of numbers and has a fitness value."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness = FitnessMax()


creator.FitnessMax = FitnessMax
creator.Individual = Individual

BUDGET_LEFTOVER_EXP = 2.0
LEFTOVER_MULTIPLIER = 1e9
PLAYER_COUNT_PENALTY = 1e9
ROLE_MISSING_PENALTY = 1e9
SINGLE_PLAYER_CAP_RATIO = 0.4
HIGH_PENALTY = 999999999
USE_SURROGATE = True
SURROGATE_THRESHOLD = 50
evaluation_count = 0


class SurrogateModel:
    def __init__(self):
        self.X: List[List[float]] = []
        self.y: List[float] = []
        self.mean = 0.0

    def update(self, candidate: List[float], fitness_value: float) -> None:
        self.X.append(candidate)
        self.y.append(fitness_value)

    def train(self) -> None:
        self.mean = float(np.mean(self.y)) if self.y else 0.0

    def evaluate(self, _candidate: List[float]) -> float:
        return self.mean


surrogate_model = SurrogateModel() if USE_SURROGATE else None


def min_bid_threshold(_manager) -> int:
    return 1


def max_bid_for_player(manager) -> float:
    budget = manager.budget
    max_total = int(manager.max_total)
    players_needed = max_total - len(manager.team)
    if players_needed <= 0:
        return budget
    base_cap = budget / players_needed
    base_max = base_cap * 2
    return min(base_max, budget)


def max_bid_possible(manager) -> float:
    budget = manager.budget
    max_total = int(manager.max_total)
    players_needed = max_total - len(manager.team)
    return budget - (players_needed - 1)


def role_weight(manager, role_name: str) -> float:
    current_count = sum(1 for p in manager.team if p.role == role_name)
    min_r, _ = manager.role_constraints[role_name]
    return 2.0 if current_count < min_r else 1.0


def common_fitness_logic(manager, bids: List[float], roles: List[str], scores: List[float], min_thr: int) -> float:
    global evaluation_count, surrogate_model
    evaluation_count += 1
    int_bids = [int(round(b)) for b in bids]
    for i, bid_value in enumerate(int_bids):
        if 0 < bid_value < min_thr:
            int_bids[i] = min_thr
    budget = manager.budget
    total_spent = sum(int_bids)
    if total_spent > budget:
        return HIGH_PENALTY
    for bid_value in int_bids:
        if bid_value > budget * SINGLE_PLAYER_CAP_RATIO:
            return HIGH_PENALTY
    leftover_budget = budget - total_spent
    max_total = int(manager.max_total)
    players_needed_local = max_total - len(manager.team)
    if leftover_budget < players_needed_local:
        return HIGH_PENALTY
    leftover_penalty = ((leftover_budget - players_needed_local) ** BUDGET_LEFTOVER_EXP) * LEFTOVER_MULTIPLIER
    chosen_count = sum(1 for v in int_bids if v >= min_thr)
    penalty = abs(chosen_count - players_needed_local) * PLAYER_COUNT_PENALTY
    role_count = {}
    for i, bid_value in enumerate(int_bids):
        if bid_value >= min_thr:
            r = roles[i]
            role_count[r] = role_count.get(r, 0) + 1
    for r, (min_r, max_r) in manager.role_constraints.items():
        current_have = sum(1 for p in manager.team if p.role == r)
        add_count = role_count.get(r, 0)
        if current_have + add_count < min_r or current_have + add_count > max_r:
            return HIGH_PENALTY
    penalty += leftover_penalty
    total_score = 0.0
    for i, bid_value in enumerate(int_bids):
        if bid_value >= min_thr:
            total_score += role_weight(manager, roles[i]) * scores[i]
    computed_fitness = penalty - total_score
    if USE_SURROGATE and surrogate_model is not None:
        surrogate_model.update(bids, computed_fitness)
        if evaluation_count % 20 == 0:
            surrogate_model.train()
        if evaluation_count >= SURROGATE_THRESHOLD:
            return 0.5 * computed_fitness + 0.5 * surrogate_model.evaluate(bids)
    return computed_fitness


def manager_strategy_pso(manager, players_not_assigned):
    budget = manager.budget
    max_total = int(manager.max_total)
    if budget <= 0 or (max_total - len(manager.team)) <= 0:
        return []
    mb_possible = max_bid_possible(manager)
    max_bid_per_player = min(max_bid_for_player(manager), mb_possible)
    if max_bid_per_player < 1:
        return []
    n = len(players_not_assigned)
    if n == 0:
        return []
    lb = np.array([0.0 for _ in range(n)], dtype=np.float64)
    ub = np.array([max_bid_per_player for _ in range(n)], dtype=np.float64)
    pids = [pl.pid for pl in players_not_assigned]
    roles = [pl.role for pl in players_not_assigned]
    scores = [score_player(pl) for pl in players_not_assigned]
    min_thr = min_bid_threshold(manager)

    def fitness_func(bids_vector: List[float]) -> float:
        return common_fitness_logic(manager, bids_vector, roles, scores, min_thr)

    best_bids, _ = pso(fitness_func, lb, ub, swarmsize=40, maxiter=80, omega=0.7, phip=1.8, phig=1.8)
    final_bids = [int(round(b)) for b in best_bids]
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value < min_thr:
            final_bids[i] = min_thr
    return [(pids[i], bid_value) for i, bid_value in enumerate(final_bids) if 0 < bid_value <= budget]


def manager_strategy_de(manager, players_not_assigned):
    if manager.budget <= 0:
        return []
    max_total = int(manager.max_total)
    if (max_total - len(manager.team)) <= 0:
        return []
    mb_possible = max_bid_possible(manager)
    max_bid_per_player = min(max_bid_for_player(manager), mb_possible)
    if max_bid_per_player < 1:
        return []
    n = len(players_not_assigned)
    if n == 0:
        return []
    roles = [pl.role for pl in players_not_assigned]
    scores = [score_player(pl) for pl in players_not_assigned]
    min_thr = min_bid_threshold(manager)

    def fitness_wrapper(bids_vector: List[float]) -> float:
        return common_fitness_logic(manager, bids_vector, roles, scores, min_thr)

    result = differential_evolution(fitness_wrapper, bounds=[(0.0, float(max_bid_per_player))] * n,
                                    strategy='best1bin', maxiter=50, popsize=15,
                                    mutation=(0.5, 1.0), recombination=0.7)
    final_bids = [int(round(b)) for b in result.x]
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value < min_thr:
            final_bids[i] = min_thr
    pids = [pl.pid for pl in players_not_assigned]
    return [(pids[i], bid_value) for i, bid_value in enumerate(final_bids) if 0 < bid_value <= manager.budget]


def manager_strategy_es(manager, players_not_assigned):
    if manager.budget <= 0:
        return []
    max_total = int(manager.max_total)
    if (max_total - len(manager.team)) <= 0:
        return []
    mb_possible = max_bid_possible(manager)
    max_bid_per_player = min(max_bid_for_player(manager), mb_possible)
    if max_bid_per_player < 1:
        return []
    n = len(players_not_assigned)
    if n == 0:
        return []
    toolbox = base.Toolbox()

    def init_value() -> float:
        return random.uniform(0.0, float(max_bid_per_player))

    def init_individual() -> Individual:
        return Individual([init_value() for _ in range(n)])

    population_ = [init_individual() for _ in range(40)]
    roles = [pl.role for pl in players_not_assigned]
    scores = [score_player(pl) for pl in players_not_assigned]
    pids = [pl.pid for pl in players_not_assigned]
    min_thr = min_bid_threshold(manager)

    def eval_es(individual: List[float]) -> Tuple[float]:
        return common_fitness_logic(manager, individual, roles, scores, min_thr),

    toolbox.register("evaluate", eval_es)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=(max_bid_per_player / 5), indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    algorithms.eaMuPlusLambda(population_, toolbox, mu=20, lambda_=40, cxpb=0.5, mutpb=0.4, ngen=30, verbose=False)
    best_ind = tools.selBest(population_, 1)[0]
    final_bids = [max(0, min(int(round(b)), int(manager.budget))) for b in best_ind]
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value < min_thr:
            final_bids[i] = min_thr
    return [(pids[i], bid_value) for i, bid_value in enumerate(final_bids) if 0 < bid_value <= manager.budget]


def resolve_competition(manager_offers, min_increment=1, max_rebids=5, trigger_gap=3):
    if not manager_offers:
        return None, 0
    manager_offers.sort(key=lambda x: x[1], reverse=True)
    if len(manager_offers) == 1:
        return manager_offers[0]
    top_man, top_bid = manager_offers[0]
    second_man, second_bid = manager_offers[1]
    diff = top_bid - second_bid
    if diff > trigger_gap:
        return top_man, top_bid
    reb_count = 0
    while reb_count < max_rebids:
        needed = second_man.max_total - len(second_man.team)
        if needed <= 0:
            break
        ratio = second_man.budget / float(needed)
        dynamic_inc = max(min_increment, int(round((top_bid - second_bid) / 2 * ratio))) + 1
        if dynamic_inc + second_bid > second_man.budget:
            break
        second_bid += dynamic_inc
        top_man, second_man = second_man, top_man
        top_bid, second_bid = second_bid, top_bid
        diff = top_bid - second_bid
        if diff > trigger_gap:
            break
        reb_count += 1
    return top_man, top_bid


def multi_manager_auction(players, managers, max_turns=30):
    not_assigned = {p.pid: p for p in players}
    turn_counter = 0
    forced_assignments = {mgr.name: [] for mgr in managers}
    overspent_assignments = {mgr.name: [] for mgr in managers}

    while turn_counter < max_turns:
        turn_counter += 1
        print(f"\n=== TURN {turn_counter}/{max_turns} ===")
        all_bids = []
        for mgr in managers:
            avail_pl = list(not_assigned.values())
            bids = mgr.decide_bids(avail_pl)
            for (pid, amt) in bids:
                if pid in not_assigned and mgr.can_buy(not_assigned[pid], amt):
                    all_bids.append((mgr, pid, amt))
        if not all_bids:
            print("No bids were made. Ending auction.")
            break
        bids_by_player = {}
        for (mgr, pid, amt) in all_bids:
            bids_by_player.setdefault(pid, []).append((mgr, amt))
        for pid, mgr_offs in bids_by_player.items():
            if len(mgr_offs) == 1:
                best_manager, best_amt = mgr_offs[0]
            else:
                best_manager, best_amt = resolve_competition(mgr_offs, min_increment=1, max_rebids=5, trigger_gap=3)
            if pid in not_assigned:
                player_obj = not_assigned[pid]
                if best_manager.can_buy(player_obj, best_amt):
                    player_obj.assigned_to = best_manager.name
                    player_obj.final_price = best_amt
                    best_manager.update_roster(player_obj, best_amt)
                    del not_assigned[pid]
        all_out = all(mgr.budget <= 0 or len(mgr.team) >= mgr.max_total for mgr in managers)
        if all_out:
            print("All managers are out of budget or have completed their rosters.")
            break

    # Complete exact role minima first. update_roster already deducts the price, so
    # do not deduct the same forced credit a second time.
    for mgr in managers:
        for role_name, (min_r, max_r) in mgr.role_constraints.items():
            current_count = sum(1 for p in mgr.team if p.role == role_name)
            missing = max(0, min_r - current_count)
            available = sorted([p for p in not_assigned.values() if p.role == role_name],
                               key=score_player, reverse=True)
            for _ in range(missing):
                if available and mgr.budget >= 1:
                    chosen = available.pop(0)
                    chosen.assigned_to = mgr.name
                    chosen.final_price = 1
                    mgr.update_roster(chosen, 1)
                    forced_assignments[mgr.name].append(chosen)
                    not_assigned.pop(chosen.pid, None)
                else:
                    overspent_assignments[mgr.name].append(
                        f"Role {role_name}: insufficient budget/players to force {missing} players")
                    break

        still_needed = mgr.max_total - len(mgr.team)
        if still_needed > 0:
            remaining_list = sorted(list(not_assigned.values()), key=score_player, reverse=True)
            for _ in range(still_needed):
                if remaining_list and mgr.budget >= 1:
                    chosen = remaining_list.pop(0)
                    # Respect role maximums while filling any residual slot.
                    _, max_r = mgr.role_constraints.get(chosen.role, (0, 0))
                    current_count = sum(1 for p in mgr.team if p.role == chosen.role)
                    if current_count >= max_r:
                        continue
                    chosen.assigned_to = mgr.name
                    chosen.final_price = 1
                    mgr.update_roster(chosen, 1)
                    forced_assignments[mgr.name].append(chosen)
                    not_assigned.pop(chosen.pid, None)
                else:
                    break
        mgr.budget = max(0, mgr.budget)

    for mgr in managers:
        mgr.forced_assignments = forced_assignments[mgr.name]
        mgr.overspent_assignments = overspent_assignments[mgr.name]
    return managers, list(players)
