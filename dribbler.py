import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict

def assign_possessor(row: pd.Series, team_A: List[Tuple[str, str]], team_B: List[Tuple[str, str]], threshold: float = 2) -> Optional[str]:
    """
    Determines which player (if any) is in possession of the ball at a given timestamp.
    
    The function computes the Euclidean distance between the ball (converted to field units)
    and each player’s position (also converted). It returns the player’s identifier (e.g., "H3" or "A5")
    if the distance is below the given threshold; otherwise, it returns None.
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

def get_player_position(row: pd.Series, player_id: str) -> np.ndarray:
    """
    Returns the (x, y) position (in field units) for the given player identifier.
    
    The conversion uses the same scale as in the assign_possessor function.
    """
    if player_id.startswith("H"):
        index = int(player_id[1:]) - 1
        x_col = f"home_{index}_x"
        y_col = f"home_{index}_y"
    elif player_id.startswith("A"):
        index = int(player_id[1:]) - 1
        x_col = f"away_{index}_x"
        y_col = f"away_{index}_y"
    else:
        raise ValueError("Invalid player identifier")
    
    return np.array([row[x_col] * 105/2, row[y_col] * 68/2])

def count_dribbles(csv_file: str, poss_threshold: float = 2, dribble_distance_threshold: float = 5) -> Dict[str, int]:
    """
    Processes the CSV tracking data to count dribble events per team.
    
    A dribble event is detected as follows:
      - When a player (dribbler) gains possession, we check for the closest opponent from the other team.
      - If an opponent is found within dribble_distance_threshold (5 meters) and the relative order is such that
        for a home player the dribbler’s x is less than the opponent’s x (or vice versa for an away player),
        a dribble event is initiated.
      - While the dribbler retains possession, if the dribbler’s x coordinate reverses relative to the opponent’s current x,
        and the distance between them remains ≤ dribble_distance_threshold, the dribble is counted.
      - If at any point the distance exceeds the threshold or possession changes, the event is aborted.
    
    Args:
        csv_file (str): Path to the CSV file with tracking data.
        poss_threshold (float, optional): Distance threshold (in field units) used to determine possession. Defaults to 2.
        dribble_distance_threshold (float, optional): Maximum allowed distance (in meters) between dribbler and defender during a dribble. Defaults to 5.
    
    Returns:
        Dict[str, int]: A dictionary mapping team identifiers ("H" for home, "A" for away) to the number of dribbles.
    """
    df: pd.DataFrame = pd.read_csv(csv_file)
    
    # Define player columns for home (team_A) and away (team_B)
    team_A: List[Tuple[str, str]] = [(f"home_{i}_x", f"home_{i}_y") for i in range(11)]
    team_B: List[Tuple[str, str]] = [(f"away_{i}_x", f"away_{i}_y") for i in range(11)]
    
    team_dribbles: Dict[str, int] = {"H": 0, "A": 0}
    prev_possessor: Optional[str] = None
    
    # Variables for tracking a dribble event
    dribble_active: bool = False
    dribble_candidate: Optional[str] = None  # the opposing player involved in the dribble
    dribbler_initial_x: Optional[float] = None
    candidate_initial_x: Optional[float] = None
    dribble_dribbler: Optional[str] = None  # the player executing the dribble
    
    # Process each frame in the CSV
    for _, row in df.iterrows():
        # Determine current possessor directly from the assign_possessor function.
        current_possessor: Optional[str] = assign_possessor(row, team_A, team_B, poss_threshold)
        
        # If possession changes, abort any ongoing dribble event.
        if current_possessor != prev_possessor:
            dribble_active = False
            dribble_candidate = None
            dribbler_initial_x = None
            candidate_initial_x = None
            dribble_dribbler = current_possessor
        
        if current_possessor is not None:
            # Get current dribbler's position.
            dribbler_pos: np.ndarray = get_player_position(row, current_possessor)
            dribbler_x: float = dribbler_pos[0]
            team = current_possessor[0]  # "H" or "A"
            opponent_team = "A" if team == "H" else "H"
            
            # If no dribble event is active, try to start one by finding a nearby opponent.
            if not dribble_active:
                candidate_id: Optional[str] = None
                candidate_distance: float = np.inf
                candidate_x: Optional[float] = None
                
                if opponent_team == "A":
                    # For a home dribbler, search among away players.
                    for i in range(11):
                        opp_id = f"A{i+1}"
                        opp_pos = get_player_position(row, opp_id)
                        distance = np.linalg.norm(dribbler_pos - opp_pos)
                        # For home, require that the dribbler starts behind the opponent.
                        if distance <= dribble_distance_threshold and distance < candidate_distance and dribbler_x < opp_pos[0]:
                            candidate_distance = distance
                            candidate_id = opp_id
                            candidate_x = opp_pos[0]
                else:
                    # For an away dribbler, search among home players.
                    for i in range(11):
                        opp_id = f"H{i+1}"
                        opp_pos = get_player_position(row, opp_id)
                        distance = np.linalg.norm(dribbler_pos - opp_pos)
                        # For away, require that the dribbler starts ahead of the opponent.
                        if distance <= dribble_distance_threshold and distance < candidate_distance and dribbler_x > opp_pos[0]:
                            candidate_distance = distance
                            candidate_id = opp_id
                            candidate_x = opp_pos[0]
                
                if candidate_id is not None:
                    # Start a dribble event.
                    dribble_active = True
                    dribble_candidate = candidate_id
                    dribbler_initial_x = dribbler_x
                    candidate_initial_x = candidate_x
                    dribble_dribbler = current_possessor
            
            else:
                # A dribble event is ongoing.
                if current_possessor != dribble_dribbler:
                    # Possession changed during the event; abort.
                    dribble_active = False
                    dribble_candidate = None
                    dribbler_initial_x = None
                    candidate_initial_x = None
                else:
                    # Retrieve the candidate's current position.
                    if dribble_candidate is not None:
                        candidate_pos = get_player_position(row, dribble_candidate)
                        candidate_x_current = candidate_pos[0]
                        # If the distance grows too large, abort the event.
                        if np.linalg.norm(dribbler_pos - candidate_pos) > dribble_distance_threshold:
                            dribble_active = False
                            dribble_candidate = None
                            dribbler_initial_x = None
                            candidate_initial_x = None
                        else:
                            # Check if the dribbler has "surpassed" the candidate.
                            if team == "H":
                                # For a home player: initially dribbler_x < candidate_initial_x; dribble succeeds if dribbler_x > candidate_x_current.
                                if dribbler_initial_x is not None and candidate_initial_x is not None:
                                    if dribbler_x > candidate_x_current:
                                        team_dribbles["H"] += 1
                                        dribble_active = False
                                        dribble_candidate = None
                                        dribbler_initial_x = None
                                        candidate_initial_x = None
                            else:
                                # For an away player: initially dribbler_x > candidate_initial_x; dribble succeeds if dribbler_x < candidate_x_current.
                                if dribbler_initial_x is not None and candidate_initial_x is not None:
                                    if dribbler_x < candidate_x_current:
                                        team_dribbles["A"] += 1
                                        dribble_active = False
                                        dribble_candidate = None
                                        dribbler_initial_x = None
                                        candidate_initial_x = None
        
        prev_possessor = current_possessor
    
    return team_dribbles

def main() -> None:
    """
    Main function to count dribbles from CSV tracking data and display the results.
    
    It reads tracking data from a CSV file, counts the dribbles per team based on the criteria,
    and then prints the results.
    """
    # Replace with your actual CSV file path.
    csv_file: str = "/home/tzikos/Desktop/jsons/val/stitched_game_3.csv"
    dribble_counts = count_dribbles(csv_file, poss_threshold=1, dribble_distance_threshold=2)
    
    print("Dribble counts by team:")
    print(f"Home (H): {dribble_counts['H']} dribbles")
    print(f"Away (A): {dribble_counts['A']} dribbles")

if __name__ == "__main__":
    main()
