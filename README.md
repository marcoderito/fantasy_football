# Fantasy Football Optimization Project

This project applies evolutionary and swarm intelligence algorithms (PSO, DE, ES) to solve an inverse optimization problem in the context of Fantasy Football. Based on historical player data,  the aim is to optimize fantasy football team composition while respecting budget and role constraints.

## Overview

This project uses evolutionary and swarm intelligence algorithms (Particle Swarm Optimization, Differential Evolution, Evolution Strategies) to optimize fantasy football team selection. Given historical player performance data, the goal is to determine the optimal composition of a fantasy football team.

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
  Generates a PDF report that includes:
  - Technical descriptions of PSO, DE, and ES algorithms and their roles in the optimization process.
  - Detailed empirical analysis such as average scores, distribution of managers’ chosen strategies, and player performance statistics visualized through multiple charts (bar charts, pie charts, and histograms).
  - Tabular summaries showing key performance indicators for each optimization strategy.
  - Clearly structured listings of the final teams selected by each manager, detailing player roles, bids, and calculated performance scores.

  - **hyperparameter_tuning.py:**  
  Performs hyper-parameter optimization for PSO, DE, and ES using a one-on-one simulated auction (1 test manager vs. 1 random rival).  
  The goal is to empirically determine parameter configurations that maximize team quality (score), minimize forced assignments, and optimize budget usage.  
  The script runs multiple configurations and records performance metrics to compare strategies under varying settings. Results are printed and can be logged for future analysis.

- **inverse_param_tuning.py:**  
  Implements inverse optimization: given a fixed target performance profile (e.g., score = 100, forced = 4, leftover = 0),  
  the script searches for the algorithm and hyper-parameters that reproduce this target when running an internal auction.  
  It uses Particle Swarm Optimization (PSO) as an outer optimizer to minimize the $\ell_1$ loss between actual and desired KPIs.  
  This enables retrospective tuning and validation of which parameter settings would likely lead to a predefined auction outcome.


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

| Id | R | Rm | Name | Team | Pv | Mv | Fm | Gf | Gs | Rp | Rc | R+ | R- | Ass | Amm | Esp | Au |
|----|---|----|------|------|----|----|----|----|----|----|----|----|----|-----|-----|-----|----|
| 2170 | P | Por | Milinkovic-Savic V. | Torino     | 25 | 6.44 | 5.72 | 0 | 29 | 4 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
|  572 | P | Por | Meret               | Napoli     | 23 | 6.35 | 5.61 | 0 | 20 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2521 | P | Por | De Gea             | Fiorentina | 23 | 6.41 | 5.59 | 0 | 25 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|  188 | P | Por | Leali              | Genoa      | 19 | 6.21 | 5.24 | 0 | 19 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| 2428 | P | Por | Sommer             | Inter      | 25 | 6.12 | 5.28 | 0 | 24 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


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
### Optional: Run Standalone Tuning Scripts

The two tuning modules — `hyperparameter_tuning.py` and `inverse_param_tuning.py` — can be executed independently of the main optimization pipeline.

They are intended for research and evaluation purposes:

- **`hyperparameter_tuning.py`** evaluates the impact of different meta-heuristic configurations on auction performance.  
- **`inverse_param_tuning.py`** finds which parameter combinations best reproduce a predefined auction outcome.

To run them from the terminal:

```bash
python hyperparameter_tuning.py
```bash
python inverse_param_tuning.py


### 3. Review the Generated Report

Open the generated `report.pdf` file to review:

- Technical descriptions of algorithms
- Empirical performance analysis charts and tables
- Final team compositions and player statistics

## Customization

- Modify optimization parameters (swarm size, iterations, mutation rates) in `optimization.py`.

## License

This project is for educational purposes. 