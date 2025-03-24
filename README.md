# Fantasy Football Optimization Project

This project applies evolutionary and swarm intelligence algorithms (PSO, DE, ES) to solve an inverse optimization problem in the context of Fantasy Football. Based on historical player data,  the aim is to optimize fantasy football team composition while respecting budget and role constraints.

## Overview

This project uses evolutionary and swarm intelligence algorithms (Particle Swarm Optimization, Differential Evolution, Evolution Strategies) to optimize fantasy football team selection as an inverse optimization problem. Given historical player performance data, the goal is to determine the optimal composition of a fantasy football team that maximizes performance metrics, while respecting constraints such as available budget and player roles.

In particular, the project provides three alternative optimization methods that can be selected individually by each team manager during the setup phase. Managers individually select from three sophisticated optimization algorithms:

- **Particle Swarm Optimization (PSO)**:  
  Performs a global search through the player-selection space, rapidly exploring diverse team compositions to identify promising initial solutions.

- **Differential Evolution (DE)**:  
  Refines the solution iteratively by exploring local neighborhoods around candidate solutions, aiming for precise optimization and improved accuracy in player selection.

- **Evolution Strategies (ES)**:  
  Utilizes adaptive mutations and selection pressure to efficiently escape local optima, ensuring diversity and robustness in final team selections.

Each manager independently chooses the preferred algorithm (PSO, DE, or ES) to construct their optimal fantasy football team. After managers have selected their algorithms, a multi-manager auction process is conducted to assign players based on generated bids, resolving conflicts and optimizing overall team compositions.

A detailed PDF report is generated after the optimization, providing insights such as technical descriptions, empirical analysis of the algorithms, visualizations of performance metrics, and finalized team rosters.

## Project Structure

- **data_loader.py:**  
  Loads and preprocesses the Fantasy Football dataset from a CSV file. It includes functionality to handle data cleaning, ensure correct data types, and manage potential data inconsistencies or missing values. It returns a structured DataFrame ready for use in the optimization phase.

- **utils.py:**  
  Contains utility functions and class definitions:
  - **Player:** Defines player attributes (e.g., ID, role, goals scored, assists, fantasy rating, etc.).
  - **Manager:** Manages fantasy football teams, budgets, role constraints, and player assignments. It also includes methods to determine available budget, validate bids, and update team rosters.
  - **score_player:** A scoring function evaluating player performances based on fantasy football metrics like goals scored, assists, and other relevant statistics.

- **optimization.py:**  
  Provides implementations for three alternative optimization strategies:
  - **Particle Swarm Optimization (PSO)**: Quickly explores a wide solution space to propose initial bidding strategies (implemented via `pyswarm` library).
  - **Differential Evolution (DE)**: Performs refined searches to precisely tune player selection (implemented via SciPy’s `differential_evolution`).
  - **Evolution Strategies (ES)**: Applies adaptive mutations for robust optimization to avoid local minima (implemented using DEAP evolutionary algorithms library).
  
  Additionally, it manages a multi-manager auction function that resolves conflicts among managers bidding for the same players, assigns players to managers according to winning bids, and handles forced assignments when constraints must be strictly enforced.

- **main.py:**  
  Main script to execute the full fantasy football optimization pipeline. It interacts with the user via terminal prompts to define project settings, including:
  - Number of fantasy managers.
  - Budget constraints and maximum players allowed per manager.
  - Minimum and maximum role constraints (Goalkeepers, Defenders, Midfielders, Forwards).
  - Choice of optimization algorithm (PSO, DE, or ES) individually for each manager.
  
  The script proceeds through the following phases:
  - Loads and preprocesses historical player performance data.
  - Initializes manager instances and player objects according to provided constraints.
  - Runs the multi-manager auction process, where managers bid for players using their selected optimization algorithm.
  - Resolves bidding conflicts and assigns players to managers based on optimized bids.
  - Generates a detailed PDF report containing:
    - Technical descriptions of algorithms used.
    - Empirical performance analysis with graphs and statistical insights.
    - Final fantasy football teams and relevant metrics for each manager.

- **report_generator.py:**  
  Generates a comprehensive and professional PDF report that includes:
  - Technical descriptions of PSO, DE, and ES algorithms and their roles in the optimization process.
  - Detailed empirical analysis such as average scores, distribution of managers’ chosen strategies, and player performance statistics visualized through multiple charts (bar charts, pie charts, and histograms).
  - Tabular summaries showing key performance indicators for each optimization strategy.
  - Clearly structured listings of the final teams selected by each manager, detailing player roles, bids, and calculated performance scores.

- **requirements.txt:**  
  Lists all Python dependencies required to run the project successfully, ensuring easy setup and reproducibility of the environment.

## Installation

Follow these steps to set up the project:

```bash
git clone https://github.com/marcoderito/fantasy_football.git
cd fantasy-football-optimization

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```
## Dataset Example

The dataset (Fantacalcio_stat.csv) should include columns similar to the following example:

| Name              | Role | Team  | Matches_Played | Goals_Scored | Assists | Fantasy_Rating | Yellow_Cards | Red_Cards |
|-------------------|------|-------|----------------|--------------|---------|----------------|--------------|-----------|
| Lautaro Martinez  | A    | Inter | 25             | 15           | 5       | 7.8            | 3            | 0         |
| Theo Hernandez    | D    | Milan | 24             | 4            | 6       | 7.2            | 4            | 0         |
| Nicolò Barella    | C    | Inter | 27             | 6            | 8       | 7.5            | 2            | 0         |
| ...               | ...  | ...   | ...            | ...          | ...     | ...            | ...          | ...       |

## Usage

### 1. Prepare the Dataset

Ensure the Fantasy Football CSV file (e.g., `Fantacalcio_stat.csv`) is placed in the project directory. Adjust column names in `data_loader.py` if necessary.

### 2. Run the Main Script

```bash
python main.py
```

The script will prompt you for inputs such as:

- Number of managers
- Budget constraints
- Role constraints (number of players per position)

After completing the optimization, the script will print team details to the console and generate a PDF report named `report.pdf`.

### 3. Review the Generated Report

Open the generated `report.pdf` file to review:

- Technical descriptions of algorithms
- Empirical performance analysis charts and tables
- Final team compositions and player statistics

## Customization

- Modify optimization parameters (swarm size, iterations, mutation rates) in `optimization.py`.
- Adapt the PDF report content and graphical visualizations in `report_generator.py`.

## Acknowledgements

This project was inspired by the Fuzzy Self-Tuning PSO paper and the course materials provided in the course zip file. Special thanks to the course instructors for their guidance and support.

## License

This project is for educational purposes. Feel free to modify and use it for your own projects or coursework.

