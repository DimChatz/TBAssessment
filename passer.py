import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict

def assign_possessor(row: pd.Series, team_A: List[Tuple[str, str]], team_B: List[Tuple[str, str]], threshold: float = 7) -> Optional[str]:
    """
    Determines which player (if any) is in possession of the ball at a given timestamp.

    This function calculates the Euclidean distance between the ball position and each player's position 
    for both teams. If a player's distance is less than the specified threshold and is the closest found, 
    the player's identifier (e.g., 'H3' or 'A5') is returned. If no player is within the threshold, returns None.

    Args:
        row (pd.Series): A row from the tracking DataFrame containing ball and player positions.
        team_A (List[Tuple[str, str]]): List of tuples containing the column names for team A's x and y positions.
        team_B (List[Tuple[str, str]]): List of tuples containing the column names for team B's x and y positions.
        threshold (float, optional): Distance threshold to determine possession. Defaults to 7.

    Returns:
        Optional[str]: The identifier of the possessing player (e.g., 'H3' or 'A5') if within threshold, otherwise None.
    """
    ball_pos = np.array([row["ball_x"] * 105 / 2, row["ball_y"] * 68 / 2])
    
    best_player: Optional[str] = None
    best_distance: float = np.inf
    
    # Check team A players
    for i, (x_col, y_col) in enumerate(team_A):
        player_pos = np.array([row[x_col] * 105 / 2, row[y_col] * 68 / 2])
        distance = np.linalg.norm(ball_pos - player_pos)
        if distance < best_distance and distance < threshold:
            best_distance = distance
            best_player = f"H{i+1}"
            
    # Check team B players
    for i, (x_col, y_col) in enumerate(team_B):
        player_pos = np.array([row[x_col] * 105 / 2, row[y_col] * 68 / 2])
        distance = np.linalg.norm(ball_pos - player_pos)
        if distance < best_distance and distance < threshold:
            best_distance = distance
            best_player = f"A{i+1}"
            
    return best_player

def count_passes(csv_file: str, threshold: float = 15) -> Dict[Tuple[str, str], int]:
    """
    Counts passes between players by processing CSV tracking data.

    A pass is detected when the player in possession changes from one player to another within the same team 
    between consecutive frames. If no player meets the threshold in a given frame, the previous possessor is retained.

    Args:
        csv_file (str): The path to the CSV file containing the tracking data.
        threshold (float, optional): Distance threshold for determining possession. Defaults to 15.

    Returns:
        Dict[Tuple[str, str], int]: A dictionary with keys as tuples (from_player, to_player) indicating a pass,
        and values as the count of passes between those players.
    """
    df = pd.read_csv(csv_file)
    
    # Define player columns for each team (assumes 11 players per team)
    team_A: List[Tuple[str, str]] = [(f"home_{i}_x", f"home_{i}_y") for i in range(11)]
    team_B: List[Tuple[str, str]] = [(f"away_{i}_x", f"away_{i}_y") for i in range(11)]
    
    pass_counts: Dict[Tuple[str, str], int] = {}  # Keys: (from_player, to_player), Value: count
    prev_possessor: Optional[str] = None

    # Iterate over each timestamp (row) in the dataset
    for _, row in df.iterrows():
        current_possessor = assign_possessor(row, team_A, team_B, threshold)
        
        # If no new possessor is detected, retain the previous possessor
        if current_possessor is None:
            current_possessor = prev_possessor
        
        # If both current and previous possessors exist and are different, and belong to the same team, count a pass
        if prev_possessor is not None and current_possessor is not None:
            if prev_possessor != current_possessor and prev_possessor[0] == current_possessor[0]:
                key = (prev_possessor, current_possessor)
                pass_counts[key] = pass_counts.get(key, 0) + 1
        
        # Update the previous possessor for the next iteration
        prev_possessor = current_possessor

    return pass_counts

def main() -> None:
    """
    Main function to count passes between players using CSV tracking data.

    This function reads the tracking data from a CSV file, counts the passes between players based on possession changes,
    and prints the individual pass counts as well as the total number of passes.

    Returns:
        None
    """
    # Replace with your actual CSV file path
    csv_file: str = "/home/tzikos/Desktop/jsons/val/stitched_game_3.csv"
    passes = count_passes(csv_file, threshold=2)
    
    print("Pass counts between players:")
    for (player_from, player_to), count in passes.items():
        print(f"{player_from} -> {player_to}: {count} passes")
    
    # Calculate and display the total passes
    total_passes: int = sum(passes.values())
    print(f"\nTotal passes: {total_passes}")

if __name__ == "__main__":
    main()
