import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict

def assign_possessor(row: pd.Series, team_A: List[Tuple[str, str]], team_B: List[Tuple[str, str]], threshold: float = 15) -> Optional[str]:
    """
    Determines which player (if any) is in possession of the ball at a given timestamp.
    
    This function calculates the Euclidean distance between the ball and each player's position.
    It returns the player's identifier (e.g., 'H3' for a home player or 'A5' for an away player)
    if the player is within the specified threshold distance. If no player is close enough,
    the function returns None.
    
    Args:
        row (pd.Series): A row from the tracking DataFrame containing ball and player positions.
        team_A (List[Tuple[str, str]]): List of tuples for home team player column names (x, y).
        team_B (List[Tuple[str, str]]): List of tuples for away team player column names (x, y).
        threshold (float, optional): Distance threshold to determine possession. Defaults to 15.
    
    Returns:
        Optional[str]: The identifier of the possessing player (e.g., 'H3' or 'A5') if within threshold; otherwise, None.
    """
    ball_pos: np.ndarray = np.array([row["ball_x"] * 105/2, row["ball_y"] * 68/2])
    
    best_player: Optional[str] = None
    best_distance: float = np.inf
    
    # Check home team players (team A)
    for i, (x_col, y_col) in enumerate(team_A):
        player_pos: np.ndarray = np.array([row[x_col] * 105/2, row[y_col] * 68/2])
        distance: float = np.linalg.norm(ball_pos - player_pos)
        if distance < best_distance and distance < threshold:
            best_distance = distance
            best_player = f"H{i+1}"
    
    # Check away team players (team B)
    for i, (x_col, y_col) in enumerate(team_B):
        player_pos: np.ndarray = np.array([row[x_col] * 105/2, row[y_col] * 68/2])
        distance: float = np.linalg.norm(ball_pos - player_pos)
        if distance < best_distance and distance < threshold:
            best_distance = distance
            best_player = f"A{i+1}"
    
    return best_player

def count_shots(csv_file: str, shot_speed_threshold: float = 5, poss_threshold: float = 15) -> Dict[str, int]:
    """
    Counts shots by processing the CSV tracking data.
    
    A shot is detected when:
      - The ball's x velocity exceeds the shot_speed_threshold:
          - For a home-team shot, the velocity must be high and positive, and the ball must be in the enemy half (ball_x > 52.5).
          - For an away-team shot, the velocity must be high and negative, and the ball must be in the enemy half (ball_x < 52.5).
      - The team that had possession in the previous frame (prev_possessor) is the shooting team.
    
    If no new possessor is identified (i.e., no player is within the threshold),
    the function retains the last known possessor.
    
    Args:
        csv_file (str): Path to the CSV file with tracking data.
        shot_speed_threshold (float, optional): The minimum x velocity (in field units per frame) required to register a shot.
            Defaults to 5.
        poss_threshold (float, optional): The distance threshold (in field units) to detect a player in possession.
            Defaults to 15.
    
    Returns:
        Dict[str, int]: A dictionary mapping player identifiers (who took shots) to the number of shots counted.
    """
    df: pd.DataFrame = pd.read_csv(csv_file)
    
    # Define player columns for home (team_A) and away (team_B)
    team_A: List[Tuple[str, str]] = [(f"home_{i}_x", f"home_{i}_y") for i in range(11)]
    team_B: List[Tuple[str, str]] = [(f"away_{i}_x", f"away_{i}_y") for i in range(11)]
    
    shot_counts: Dict[str, int] = {}
    prev_possessor: Optional[str] = None
    prev_ball_x: Optional[float] = None
    shot_in_progress: bool = False  # Flag to avoid counting the same shot over consecutive frames

    # Process each frame in the CSV
    for _, row in df.iterrows():
        # Determine the current possessor; if none, retain the previous possessor.
        current_possessor: Optional[str] = assign_possessor(row, team_A, team_B, poss_threshold)
        if current_possessor is None:
            current_possessor = prev_possessor
        
        # Convert the ball's normalized x-coordinate to field units.
        ball_x: float = row["ball_x"] * 105/2
        
        # Initialize previous ball position on the first frame.
        if prev_ball_x is None:
            prev_ball_x = ball_x
            prev_possessor = current_possessor
            continue
        
        # Compute ball's x velocity between frames.
        vel_x: float = ball_x - prev_ball_x
        
        # For a home-team shot:
        #   - The ball must be moving to the right with high velocity.
        #   - The previous possessor must be from the home team.
        #   - The ball must be in the enemy half (ball_x > 0).
        if (vel_x > shot_speed_threshold and 
            prev_possessor is not None and prev_possessor.startswith("H") and 
            not shot_in_progress and ball_x > 0):
            shot_counts[prev_possessor] = shot_counts.get(prev_possessor, 0) + 1
            shot_in_progress = True
        
        # For an away-team shot:
        #   - The ball must be moving to the left with high velocity.
        #   - The previous possessor must be from the away team.
        #   - The ball must be in the enemy half (ball_x < 0).
        elif (vel_x < -shot_speed_threshold and 
              prev_possessor is not None and prev_possessor.startswith("A") and 
              not shot_in_progress and ball_x < 0):
            shot_counts[prev_possessor] = shot_counts.get(prev_possessor, 0) + 1
            shot_in_progress = True
        
        # Reset the shot flag once the velocity falls below the threshold.
        if abs(vel_x) <= shot_speed_threshold:
            shot_in_progress = False
        
        # Update values for the next frame.
        prev_ball_x = ball_x
        prev_possessor = current_possessor
        
    return shot_counts

def main() -> None:
    """
    Main function to count shots from CSV tracking data and display the results.
    
    This function reads tracking data from a CSV file, processes the data to count shots by players
    based on shot speed and possession, and then prints the shot counts for each player along with
    the total number of shots.
    
    Returns:
        None
    """
    # Replace with your actual CSV file path.
    csv_file: str = "/home/tzikos/Desktop/jsons/train/stitched_game_0.csv"
    shots: Dict[str, int] = count_shots(csv_file, shot_speed_threshold=2, poss_threshold=7)
    
    print("Shot counts by players:")
    for player, count in shots.items():
        print(f"{player}: {count} shots")
    
    total_shots: int = sum(shots.values())
    print(f"\nTotal shots: {total_shots}")

if __name__ == "__main__":
    main()