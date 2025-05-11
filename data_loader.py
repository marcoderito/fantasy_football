"""
Data Loader & Pre‑processing

Responsibilities:
  1. Read raw CSV / JSON player datasets
  2. Normalise column names, convert numeric fields
  3. Build Player objects with expected attributes
  4. Provide helper to instantiate default Manager objects

All I/O isolated here so the optimisation code remains storage‑agnostic.

Author: Marco De Rito
"""

# Import necessary modules and functions
import pandas as pd


def load_data(file_path):
    """
    Loads and cleans the Fantasy Football players dataset.
    This function reads a CSV file and cleans the data for further analysis.
    """
    try:
        # Try to load the CSV file using the specified delimiter and encoding.
        df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
    except FileNotFoundError:
        # If the file is not found, print an error message and return None.
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.ParserError:
        # If there's a parsing error, print an error message and return None.
        print(f"Error: Problem parsing the file {file_path}. Check its format.")
        return None

    # Remove whitespace from column names.
    df.columns = df.columns.str.strip()

    # Expected columns (adapt based on the actual CSV file)
    # Example: Name, Role, Goals_Scored, Assists, etc.
    # This dictionary maps the original column names to the desired names.
    expected_columns = {
        'Name': 'Name',
        'R': 'Role',
        'Team': 'Team',
        'Pv': 'Matches_Played',
        'Mv': 'Rating',
        'Fm': 'Fantasy_Rating',
        'Gf': 'Goals_Scored',
        'Ass': 'Assists',
        'Gs': 'Goals_Conceded',
        'Amm': 'Yellow_Cards',
        'Esp': 'Red_Cards',
        'Rp': 'Penalties_Scored',
        'Rc': 'Penalties_Saved'
    }

    # Check if the required columns exist in the dataset.
    missing_columns = [col for col in expected_columns.keys() if col not in df.columns]
    if missing_columns:
        # If some columns are missing, print an error message and return None.
        print(f"Error: Missing columns in the dataset: {missing_columns}")
        return None

    # Select only the columns of interest and rename them according to our mapping.
    df = df[list(expected_columns.keys())].rename(columns=expected_columns)
    # Replace any missing values with 0.
    df.fillna(0, inplace=True)

    return df
