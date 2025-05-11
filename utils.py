"""
Utility Functions

Miscellaneous helpers shared across the project:
  • `score_player(player)` computes technical score from stats
  • Random‑seed setter for reproducibility
  • Pretty‑print helpers for console summaries
  • Small maths helpers (e.g., safe division)

Kept intentionally lightweight.

Author: Marco De Rito
"""

# Player Class

class Player:
    """
    Lightweight fantasy‑player object.

    Fields
    ------
    pid, name, role           : identity data
    goals_scored, assists     : positive stats  (int, default 0)
    yellow_cards, red_cards   : penalties       (int, default 0)
    rating            : base vote       (float, default 6.0)
    penalties_scored          : spot‑kicks made (int, default 0)

    Runtime‑only
    ------------
    assigned_to : str | None  – manager name once bought
    final_price : float       – credits paid at auction
    """
    def __init__(self, pid, name, role,
                 goals_scored=0, assists=0, yellow_cards=0, red_cards=0,
                 rating=6.0, penalties_scored=0, matches_played=0, goals_conceded=0,
                 penalties_saved=0):
        # Initialize the player with basic stats and default values
        self.pid = pid  # Player ID
        self.name = name  # Player's name
        self.role = role  # Player's role/position (e.g., 'P', 'D', etc.)
        self.goals_scored = goals_scored  # Number of goals scored
        self.assists = assists  # Number of assists made
        self.yellow_cards = yellow_cards  # Number of yellow cards received
        self.red_cards = red_cards  # Number of red cards received
        self.penalties_scored = penalties_scored  # Number of penalties scored
        self.matches_played = matches_played
        self.goals_conceded = goals_conceded
        self.penalties_saved = penalties_saved
        self.rating = rating

        self.assigned_to = None  # Which manager/team the player is assigned to (if any)
        self.final_price = 0.0  # Final price paid to acquire the player

    def __repr__(self):
        # Return a string representation of the player for debugging
        return f"<Player {self.name} ({self.role})>"


# Manager Class

class Manager:
    def __init__(self, name, budget, role_constraints, max_total, strategy='pso'):
        """
        Initialize the Manager.

        Parameters:
          - name: The name of the manager.
          - budget: The available credits for the manager.
          - role_constraints: A dictionary with role limits, e.g. {'P': (3, 3), 'D': (4, 4), 'C': (4, 4), 'A': (4, 4)}.
                              Each tuple contains (min_required, max_allowed).
          - max_total: The maximum number of players allowed in the team.
          - strategy: The bidding strategy to use ('pso', 'de', or 'es').
        """
        self.name = name  # Manager's name
        self.budget = budget  # Manager's available budget
        self.role_constraints = role_constraints  # Constraints for each player role
        self.max_total = max_total  # Maximum players allowed in the team
        self.team = []  # List to store the team players
        self.strategy = strategy  # Selected bidding strategy

    def can_buy(self, player, bid_amount):
        """
        Check if the manager can purchase the given player with the bid_amount.

        Conditions:
          - The bid_amount must not exceed the current budget.
          - The team must not be already full.
          - The player's role must be managed by this manager (exists in role_constraints).
          - The team must not exceed the maximum allowed players for that role.
          - After purchase, the remaining budget must be enough to buy at least
            (missing_players - 1) credits for the rest of the players.
        """
        # Check if the bid amount is higher than the available budget
        if bid_amount > self.budget:
            return False

        # Check if the team is already full
        if len(self.team) >= self.max_total:
            return False

        # Check if the player's role is valid for this manager
        if player.role not in self.role_constraints:
            return False

        # Get the minimum and maximum allowed players for the player's role
        min_required, max_allowed = self.role_constraints[player.role]
        # Count the number of players in the team with the same role
        current_count = sum(p.role == player.role for p in self.team)
        if current_count >= max_allowed:
            return False

        # Check if the remaining budget after bidding is sufficient for the remaining players
        players_missing = self.max_total - len(self.team)
        if self.budget - bid_amount < players_missing - 1:
            return False

        # If all checks pass, the manager can buy the player
        return True

    def update_roster(self, player, price):
        """
        Add the player to the team and reduce the budget by the given price.
        """
        self.team.append(player)  # Add player to the team list
        self.budget -= price  # Deduct the purchase price from the budget

    def decide_bids(self, unassigned_players):
        """
        Decide on the bid amounts for the players that are not yet assigned,
        using the selected bidding strategy.

        Parameters:
          - unassigned_players: A list of players that have not been assigned to any team.

        Returns:
          A list of bids determined by the chosen strategy.
        """
        # If there are no players left to assign, return an empty list
        if not unassigned_players:
            return []

        # Importing the optimization strategies from the optimization module
        from optimization import (
            manager_strategy_pso,
            manager_strategy_de,
            manager_strategy_es
        )

        # Choose the bidding strategy based on the manager's configuration
        if self.strategy == 'pso':
            return manager_strategy_pso(self, unassigned_players)
        elif self.strategy == 'de':
            return manager_strategy_de(self, unassigned_players)
        elif self.strategy == 'es':
            return manager_strategy_es(self, unassigned_players)
        else:
            # Default to PSO if the strategy is unknown
            return manager_strategy_pso(self, unassigned_players)


# Function to Calculate Player Score

def score_player(player):
    """
    Compute the player's score based on various performance metrics.

  Scoring breakdown:
      - Goals Scored: +0.5 per goal
      - Assists: +0.2 per assist
      - Yellow Cards: -0.05 per card
      - Red Cards: -0.1 per card
      - Rating: +0.2 per rating
      - Penalties Scored: +0.2 per penalty
      - Goals Conceded: -0.5 per goal
      - Penalties Saved: +0.5 per save
      - Matches Played: +0.5 per match

    Parameters:
      - player: The Player object for which to calculate the score.

    Returns:
      A numerical value representing the player's overall score.
    """
    # Retrieve player statistics using getattr to provide default values if not set
    goals = getattr(player, 'goals_scored', 0)
    assists = getattr(player, 'assists', 0)
    yellow_cards = getattr(player, 'yellow_cards', 0)
    red_cards = getattr(player, 'red_cards', 0)
    rating = getattr(player, 'rating', 6.0)
    penalties = getattr(player, 'penalties_scored', 0)
    goals_conceded = getattr(player, 'goals_conceded', 0)
    penalties_saved = getattr(player, 'penalties_saved', 0)
    matches_played = getattr(player, 'matches_played', 0)

    # Calculate the score based on the weighted metrics
    score = ((0.5 * goals) + (0.2 * assists) - (0.05 * yellow_cards) - (0.1 * red_cards) + (0.2 * rating) +
             (0.2 * penalties)) - (0.5 * goals_conceded) + (0.5 * penalties_saved) + (0.5 * matches_played)
    return score
