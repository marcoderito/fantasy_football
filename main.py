# main.py

# Import necessary modules and functions
from data_loader import load_data
from utils import Player, Manager, score_player
from optimization import multi_manager_auction
from report_generator import generate_pdf_report


def main():
    # Load the dataset from the CSV file
    df = load_data("Fantacalcio_stat.csv")
    if df is None:
        # If the data failed to load, we simply return and exit
        return

    # Sort the dataframe by "Goals_Scored" in de            scending order
    df.sort_values(by="Goals_Scored", ascending=False, inplace=True)

    # Choose how many players to consider from the dataset
    num_players = 400
    df = df.head(num_players)

    # ----------------- General Settings -----------------
    print("\n--- General Settings ---")
    num_managers = int(input("Enter the number of managers: "))
    budget_input = float(input("Enter the budget for each manager: "))
    max_total_input = int(input("Enter the maximum number of players per manager: "))

    # ----------------- Role Constraints -----------------
    print("\n--- Role Constraints ---")
    # Note: 'P' = Goalkeepers, 'D' = Defenders, 'C' = Midfielders, 'A' = Forwards
    min_gk = int(input("Enter the minimum number of Goalkeepers (P): "))
    max_gk = int(input("Enter the maximum number of Goalkeepers (P): "))
    min_def = int(input("Enter the minimum number of Defenders (D): "))
    max_def = int(input("Enter the maximum number of Defenders (D): "))
    min_mid = int(input("Enter the minimum number of Midfielders (C): "))
    max_mid = int(input("Enter the maximum number of Midfielders (C): "))
    min_fw = int(input("Enter the minimum number of Forwards (A): "))
    max_fw = int(input("Enter the maximum number of Forwards (A): "))

    # Create a list to store all managers
    managers = []
    for i in range(num_managers):
        print(f"\n--- Manager {i + 1} ---")
        manager_name = f"Manager_{i + 1}"
        # Ask the user to pick a strategy (pso, de, or es)
        strategy_choice = input(f"Choose the algorithm for {manager_name} (pso/de/es): ").strip().lower()

        # Set up role constraints for this manager
        role_constraints = {
            "P": (min_gk, max_gk),
            "D": (min_def, max_def),
            "C": (min_mid, max_mid),
            "A": (min_fw, max_fw)
        }

        # Initialize the Manager object
        manager_obj = Manager(
            name=manager_name,
            budget=budget_input,
            role_constraints=role_constraints,
            max_total=max_total_input,
            strategy=strategy_choice
        )
        managers.append(manager_obj)

    # Create Player objects for each row in the dataframe
    players = []
    for idx, row in df.iterrows():
        new_player = Player(
            pid=idx,
            name=row.get("Name", f"Player_{idx}"),
            role=row.get("Role", "N/A"),
            goals_scored=row.get("Goals_Scored", 0),
            assists=row.get("Assists", 0),
            yellow_cards=row.get("Yellow_Cards", 0),
            red_cards=row.get("Red_Cards", 0),
            fantasy_rating=row.get("Fantasy_Rating", 6.0),
            penalties_scored=row.get("Penalties_Scored", 0)
        )
        players.append(new_player)

    # Run the multi-manager auction simulation
    managers_after, players_after = multi_manager_auction(players, managers, max_turns=100)

    # Print each manager's final roster and score for each player
    for mgr in managers_after:
        total_spent = sum(pl.final_price for pl in mgr.team)
        print(f"\n=== {mgr.name} === (Remaining Budget: {mgr.budget:.1f} / Spent: {total_spent:.1f})")
        for pl in mgr.team:
            pl_score = score_player(pl)
            print(f"  - {pl.name:<20} Role={pl.role} Price={pl.final_price:.1f} Score={pl_score:.2f}")

    # Generate a technical PDF report with metrics and team information
    generate_pdf_report(managers_after, filename="report.pdf")
    print("\nThe PDF report 'report.pdf' has been generated.")


# Blank line here for PEP 8 compliance (E305)
if __name__ == "__main__":
    main()
