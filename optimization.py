# Import the required libraries and modules
import numpy as np
import random
from typing import List, Tuple

# DEAP is used for evolutionary algorithms
from deap import base, creator, tools, algorithms
# pyswarm for Particle Swarm Optimization (PSO)
from pyswarm import pso
# differential_evolution from SciPy for another optimization strategy
from scipy.optimize import differential_evolution


def to_float(value):
    """
    Convert input 'value' to a float.

    - If it's a NumPy array with a single element, get that element.
    - If it's a tuple or list, take the first element.
    - Otherwise, just convert directly.

    This is useful because sometimes values come in weird formats.
    """
    # If it's a NumPy array, use .item() to get the single element
    if isinstance(value, np.ndarray):
        return float(value.item())
    # If it's a tuple or list, use the first element
    if isinstance(value, (tuple, list)):
        return float(value[0])
    # Otherwise, just cast to float
    return float(value)


##############################################
# EXPLICITLY DEFINED DEAP CLASSES
##############################################

# Define a custom fitness class for maximization problems
class FitnessMax(base.Fitness):
    """Fitness class for maximization problems."""
    weights = (1.0,)


# Define an Individual class that is just a list with a fitness attribute.
class Individual(list):
    """Individual which holds a list of numbers and has a fitness value."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness = FitnessMax()


# Register the custom classes in the DEAP creator module.
creator.FitnessMax = FitnessMax
creator.Individual = Individual

##############################################
# GLOBAL PARAMETERS
##############################################

# These constants are used in the fitness calculations and bidding logic.
BUDGET_LEFTOVER_EXP = 2.0
LEFTOVER_MULTIPLIER = 1e9
PLAYER_COUNT_PENALTY = 1e9
ROLE_MISSING_PENALTY = 1e9
SINGLE_PLAYER_CAP_RATIO = 0.4
HIGH_PENALTY = 999999999

# Variables to control the surrogate model behavior
USE_SURROGATE = True
SURROGATE_THRESHOLD = 50
evaluation_count = 0


##############################################
# SURROGATE MODEL
##############################################

class SurrogateModel:
    """
    A simple surrogate model that calculates the average of all observed fitness values.
    This is used to speed up evaluations after many runs.
    """

    def __init__(self):
        self.X: List[List[float]] = []  # store candidate solutions
        self.y: List[float] = []  # store fitness values
        self.mean = 0.0

    def update(self, candidate: List[float], fitness_value: float) -> None:
        # Save the candidate and its fitness
        self.X.append(candidate)
        self.y.append(fitness_value)

    def train(self) -> None:
        # Calculate the average fitness if there are any recorded values
        if self.y:
            self.mean = float(np.mean(self.y))
        else:
            self.mean = 0.0

    def evaluate(self, _candidate: List[float]) -> float:
        # Simply return the mean fitness
        return self.mean


# Initialize the surrogate model if it is being used
surrogate_model = SurrogateModel() if USE_SURROGATE else None


##############################################
# HELPER FUNCTIONS
##############################################

def min_bid_threshold(_manager) -> int:
    """
    Returns the minimum allowed bid, which is 1.
    """
    return 1


def max_bid_for_player(manager) -> float:
    """
    Compute the maximum bid for a single player based on the remaining budget and needed players.
    This uses a base cap calculation and then doubles it.
    """
    budget = to_float(manager.budget)
    max_total = int(to_float(manager.max_total))
    players_needed = max_total - len(manager.team)
    if players_needed <= 0:
        return budget
    base_cap = budget / players_needed  # average budget per player
    base_max = base_cap * 2
    # Make sure not to bid more than the available budget
    return min(base_max, budget)


def max_bid_possible(manager) -> float:
    """
    Returns the highest possible bid ensuring that at least 1 credit is kept for each remaining player.
    """
    budget = to_float(manager.budget)
    max_total = int(to_float(manager.max_total))
    players_needed = max_total - len(manager.team)
    return budget - (players_needed - 1)


def role_weight(manager, role_name: str) -> float:
    """
    Return a weight factor for a role.
    If the manager's team doesn't meet the minimum for that role, return 2.0,
    otherwise 1.0. This gives extra importance to under-represented roles.
    """
    current_count = sum(1 for p in manager.team if p.role == role_name)
    min_r, _ = manager.role_constraints[role_name]
    return 2.0 if current_count < min_r else 1.0


##############################################
# PLAYER SCORING FUNCTION
##############################################

# Try to import an external scoring function if available.
try:
    from utils import score_player
except ImportError:
    # If not available, use this simple scoring function.
    def score_player(_player) -> float:
        """
        Calculate a score for a player.
        For goalkeepers (role 'P'), it factors in Goals Conceded and Penalties Saved.
        For others, it just returns a constant score.
        """
        if _player.role == 'P':
            return (10.0 - float(getattr(_player, 'Goals_Conceded', 0))
                    + 3.0 * float(getattr(_player, 'Penalties_Saved', 0)))
        else:
            return 5.0


##############################################
# COMMON FITNESS CALCULATION FUNCTION
##############################################

def common_fitness_logic(manager,
                         bids: List[float],
                         roles: List[str],
                         scores: List[float],
                         min_thr: int) -> float:
    """
    Calculate the fitness of a set of bids. It checks:
      - Total spending does not exceed the budget.
      - Each bid is above the minimum threshold.
      - The leftover budget is enough for the required players.
      - Role constraints are respected.
    Finally, it adjusts the fitness with penalties and scores.
    """
    global evaluation_count, surrogate_model
    evaluation_count += 1

    # Round bids to nearest integers
    int_bids = [int(round(b)) for b in bids]
    # Ensure that any positive bid below the threshold is bumped up
    for i, bid_value in enumerate(int_bids):
        if 0 < bid_value < min_thr:
            int_bids[i] = min_thr

    # Check if total spent is over budget
    budget = to_float(manager.budget)
    total_spent = sum(int_bids)
    if total_spent > budget:
        return HIGH_PENALTY

    # Check if any individual bid is too high (more than a fixed ratio of the budget)
    for bid_value in int_bids:
        if bid_value > budget * SINGLE_PLAYER_CAP_RATIO:
            return HIGH_PENALTY

    # Calculate the budget left after the bids
    leftover_budget = budget - total_spent
    max_total = int(to_float(manager.max_total))
    players_needed_local = max_total - len(manager.team)
    # If leftover is too little for the remaining players, penalize hard
    if leftover_budget < players_needed_local:
        return HIGH_PENALTY

    # Compute a penalty for unused budget (leftover)
    leftover_penalty = ((leftover_budget - players_needed_local) ** BUDGET_LEFTOVER_EXP) * LEFTOVER_MULTIPLIER
    # Count how many bids meet the minimum threshold
    chosen_count = sum(1 for v in int_bids if v >= min_thr)
    penalty = abs(chosen_count - players_needed_local) * PLAYER_COUNT_PENALTY

    # Check role requirements: count bids for each role
    role_count = {}
    for i, bid_value in enumerate(int_bids):
        if bid_value >= min_thr:
            r = roles[i]
            role_count[r] = role_count.get(r, 0) + 1

    # Verify each role's constraints against current team and bids
    for r, (min_r, max_r) in manager.role_constraints.items():
        current_have = sum(1 for p in manager.team if p.role == r)
        add_count = role_count.get(r, 0)
        if current_have + add_count < min_r or current_have + add_count > max_r:
            return HIGH_PENALTY

    penalty += leftover_penalty

    # Calculate the total score from each bid
    total_score = 0.0
    for i, bid_value in enumerate(int_bids):
        if bid_value >= min_thr:
            w = role_weight(manager, roles[i])
            total_score += w * scores[i]

    computed_fitness = penalty - total_score

    # Use the surrogate model if activated and if we've evaluated enough candidates
    if USE_SURROGATE and surrogate_model is not None:
        surrogate_model.update(bids, computed_fitness)
        if evaluation_count % 20 == 0:
            surrogate_model.train()
        if evaluation_count >= SURROGATE_THRESHOLD:
            surrogate_val = surrogate_model.evaluate(bids)
            return 0.5 * computed_fitness + 0.5 * surrogate_val

    return computed_fitness


##############################################
# PSO STRATEGY (Particle Swarm Optimization)
##############################################

def manager_strategy_pso(manager, players_not_assigned):
    """
    Use PSO to decide the bids for each available player.
    It creates lower and upper bounds for the bid values and optimizes the fitness function.
    """
    # Convert budget and total players to numbers
    budget = to_float(manager.budget)
    max_total = int(to_float(manager.max_total))
    if budget <= 0 or (max_total - len(manager.team)) <= 0:
        return []

    # Get maximum bid possible per player
    mb_possible = max_bid_possible(manager)
    max_bid_per_player = to_float(min(max_bid_for_player(manager), mb_possible))
    if max_bid_per_player < 1:
        return []

    n = len(players_not_assigned)
    if n == 0:
        return []

    # Create lower and upper bounds for each bid (0 to max_bid_per_player)
    lb = np.array([0.0 for _ in range(n)], dtype=np.float64)
    ub = np.array([max_bid_per_player for _ in range(n)], dtype=np.float64)

    # Get player IDs, roles, and scores for the players not yet assigned
    pids = [pl.pid for pl in players_not_assigned]
    roles = [pl.role for pl in players_not_assigned]
    scores = [score_player(pl) for pl in players_not_assigned]
    min_thr = min_bid_threshold(manager)

    # Define the fitness function to be minimized
    def fitness_func(bids_vector: List[float]) -> float:
        return common_fitness_logic(manager, bids_vector, roles, scores, min_thr)

    # Run PSO to find the best bids
    best_bids, _ = pso(
        fitness_func, lb, ub,
        swarmsize=40,
        maxiter=80,
        omega=0.7,
        phip=1.8,
        phig=1.8
    )

    # Round the bids and ensure they meet the minimum threshold
    final_bids = [int(round(b)) for b in best_bids]
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value < min_thr:
            final_bids[i] = min_thr

    # Prepare the result list: (player id, bid value)
    results = []
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value <= budget:
            results.append((pids[i], bid_value))
    return results


##############################################
# DE STRATEGY (Differential Evolution)
##############################################

def manager_strategy_de(manager, players_not_assigned):
    """
    Uses Differential Evolution to optimize bids.
    Very similar to PSO but with a different optimization technique.
    """
    if to_float(manager.budget) <= 0:
        return []
    max_total = int(to_float(manager.max_total))
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

    # Run differential evolution
    result = differential_evolution(
        fitness_wrapper,
        [(0.0, float(max_bid_per_player))] * n,
        strategy='best1bin',
        maxiter=50,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7
    )

    best_bids = result.x
    final_bids = [int(round(b)) for b in best_bids]
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value < min_thr:
            final_bids[i] = min_thr

    pids = [pl.pid for pl in players_not_assigned]
    results = []
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value <= to_float(manager.budget):
            results.append((pids[i], bid_value))
    return results


##############################################
# ES STRATEGY (Evolution Strategies using DEAP)
##############################################

def manager_strategy_es(manager, players_not_assigned):
    """
    Uses an Evolution Strategy (ES) to decide on the bids.
    This approach evolves a population of bid vectors over several generations.
    """
    if to_float(manager.budget) <= 0:
        return []
    max_total = int(to_float(manager.max_total))
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

    # Function to create an initial bid value for one player randomly
    def init_value() -> float:
        return random.uniform(0.0, float(max_bid_per_player))

    # Function to create an individual (a bid vector for all players)
    def init_individual() -> Individual:
        return Individual([init_value() for _ in range(n)])

    # Create an initial population of individuals
    population_ = [init_individual() for _ in range(40)]
    roles = [pl.role for pl in players_not_assigned]
    scores = [score_player(pl) for pl in players_not_assigned]
    pids = [pl.pid for pl in players_not_assigned]
    min_thr = min_bid_threshold(manager)

    # Fitness evaluation for an individual bid vector
    def eval_es(individual: List[float]) -> Tuple[float]:
        fitness_val = common_fitness_logic(manager, individual, roles, scores, min_thr)
        # Penalize heavily if the fitness is at the HIGH_PENALTY level
        if fitness_val == HIGH_PENALTY:
            return -HIGH_PENALTY,
        return fitness_val,

    # Register genetic operators in the toolbox
    toolbox.register("evaluate", eval_es)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=(max_bid_per_player / 5), indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the evolutionary algorithm for a number of generations
    algorithms.eaMuPlusLambda(
        population_, toolbox,
        mu=40, lambda_=80,
        cxpb=0.5,
        mutpb=0.3,
        ngen=40,
        verbose=False
    )

    # Pick the best individual from the final population
    best_ind = tools.selBest(population_, k=1)[0]
    final_bids = [int(round(x)) for x in best_ind]
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value < min_thr:
            final_bids[i] = min_thr

    results = []
    for i, bid_value in enumerate(final_bids):
        if 0 < bid_value <= to_float(manager.budget):
            results.append((pids[i], bid_value))
    return results


##############################################
# COMPETITION RESOLUTION
##############################################

def resolve_competition(manager_offers: List[Tuple], min_increment=1, max_rebids=5, trigger_gap=3) -> Tuple:
    """
    Given multiple offers (bids) from different managers for the same player,
    decide which offer wins.

    It sorts the offers, and if the top two bids are too close (within the trigger gap),
    it allows for a series of rebids up to a maximum number.
    """
    if not manager_offers:
        return None, 0

    # Sort offers by bid amount, highest first
    manager_offers.sort(key=lambda x: x[1], reverse=True)
    if len(manager_offers) == 1:
        return manager_offers[0]

    top_man, top_bid = manager_offers[0]
    second_man, second_bid = manager_offers[1]
    diff = top_bid - second_bid
    if diff > trigger_gap:
        return top_man, top_bid

    # If bids are close, start a rebidding process
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

        # Swap the top and second bids to recheck the gap
        top_man, second_man = second_man, top_man
        top_bid, second_bid = second_bid, top_bid

        diff = top_bid - second_bid
        if diff > trigger_gap:
            break
        reb_count += 1

    return top_man, top_bid


##############################################
# MULTI-MANAGER AUCTION FUNCTION
##############################################

def multi_manager_auction(players, managers, max_turns=30):
    """
    This is the main auction loop where:
      - Each manager places bids on available players.
      - Bids are collected and conflicts resolved.
      - Players are assigned to managers based on winning bids.

    The auction continues for a set number of turns or until no more bids can be made.
    """
    # Dictionary of players that have not yet been assigned
    not_assigned = {p.pid: p for p in players}
    turn_counter = 0

    # To track forced assignments when role requirements are not met
    forced_assignments = {mgr.name: [] for mgr in managers}
    overspent_assignments = {mgr.name: [] for mgr in managers}

    while turn_counter < max_turns:
        turn_counter += 1
        print(f"\n=== TURN {turn_counter}/{max_turns} ===")
        all_bids = []

        # Each manager decides on bids for the available players
        for mgr in managers:
            avail_pl = list(not_assigned.values())
            bids = mgr.decide_bids(avail_pl)
            for (pid, amt) in bids:
                if pid in not_assigned and mgr.can_buy(not_assigned[pid], amt):
                    all_bids.append((mgr, pid, amt))

        if not all_bids:
            print("No bids were made. Ending auction.")
            break

        # Group bids by player
        bids_by_player = {}
        for (mgr, pid, amt) in all_bids:
            bids_by_player.setdefault(pid, []).append((mgr, amt))

        # Resolve bids for each player
        for pid, mgr_offs in bids_by_player.items():
            if len(mgr_offs) == 1:
                best_manager, best_amt = mgr_offs[0]
            else:
                best_manager, best_amt = resolve_competition(
                    mgr_offs, min_increment=1, max_rebids=5, trigger_gap=3
                )
            if pid in not_assigned:
                player_obj = not_assigned[pid]
                if best_manager.can_buy(player_obj, best_amt):
                    player_obj.assigned_to = best_manager.name
                    player_obj.final_price = best_amt
                    best_manager.update_roster(player_obj, best_amt)
                    del not_assigned[pid]

        # Check if all managers have either spent their budget or filled their team
        all_out = True
        for mgr in managers:
            if mgr.budget > 0 and len(mgr.team) < mgr.max_total:
                all_out = False
                break

        if all_out:
            print("All managers are out of budget or have completed their rosters.")
            break

    # Forced assignments for managers who haven't met role requirements or team size
    for mgr in managers:
        for role_name, (min_r, max_r) in mgr.role_constraints.items():
            current_count = sum(1 for p in mgr.team if p.role == role_name)
            if current_count < min_r:
                missing = min_r - current_count
                available = [p for p in not_assigned.values() if p.role == role_name]
                available.sort(key=lambda x: score_player(x), reverse=True)
                for _ in range(missing):
                    if available and mgr.budget >= 1:
                        chosen = available.pop(0)
                        chosen.assigned_to = mgr.name
                        chosen.final_price = 1
                        mgr.update_roster(chosen, 1)
                        forced_assignments[mgr.name].append(chosen)
                        if chosen.pid in not_assigned:
                            del not_assigned[chosen.pid]
                        mgr.budget -= 1
                    else:
                        overspent_assignments[mgr.name].append(
                            f"Role {role_name}: insufficient budget to force {missing} players"
                        )
                        break

        # Fill up remaining slots if possible
        still_needed = mgr.max_total - len(mgr.team)
        if still_needed > 0:
            forced_possible = int(min(still_needed, mgr.budget))
            if forced_possible < still_needed:
                overspent_assignments[mgr.name].append(
                    f"Insufficient budget: only {forced_possible} forced players out of {still_needed} missing"
                )
            remaining_list = list(not_assigned.values())
            remaining_list.sort(key=lambda x: score_player(x), reverse=True)
            for _ in range(forced_possible):
                if remaining_list and mgr.budget >= 1:
                    chosen = remaining_list.pop(0)
                    chosen.assigned_to = mgr.name
                    chosen.final_price = 1
                    mgr.update_roster(chosen, 1)
                    forced_assignments[mgr.name].append(chosen)
                    if chosen.pid in not_assigned:
                        del not_assigned[chosen.pid]
                    mgr.budget -= 1
                else:
                    overspent_assignments[mgr.name].append(
                        "Forced player missing (insufficient budget)"
                    )
                    break
        mgr.budget = 0

    # Save the forced and overspent assignments in each manager
    for mgr in managers:
        mgr.forced_assignments = forced_assignments[mgr.name]
        mgr.overspent_assignments = overspent_assignments[mgr.name]

    return managers, list(players)
