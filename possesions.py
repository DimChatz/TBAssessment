import pandas as pd
import numpy as np
from typing import Tuple

def compute_possession(csv_file: str, threshold: int = 2) -> Tuple[float, float]:
    """
    Compute the possession percentage for two teams based on tracking data.

    This function reads tracking data from a CSV file and calculates which team is in possession
    of the ball at each timestamp. Possession is determined by finding the closest player from each team
    to the ball. If both teams are too far from the ball (i.e. beyond the threshold), the last assigned
    possession is used. The final possession percentages for both teams are then computed.

    Args:
        csv_file (str): The path to the CSV file containing the tracking data.
        threshold (int, optional): The distance threshold (in the same units as the field dimensions)
            beyond which the ball is considered too far for either team to claim possession.
            Defaults to 1.

    Returns:
        Tuple[float, float]: A tuple where the first element is the possession percentage for Team A (Home)
        and the second element is the possession percentage for Team B (Away).
    """
    # Load tracking data from CSV
    df = pd.read_csv(csv_file)

    # Define the player column pairs for each team (assumes 11 players per team).
    team_A = [(f"home_{i}_x", f"home_{i}_y") for i in range(11)]
    team_B = [(f"away_{i}_x", f"away_{i}_y") for i in range(11)]

    possession_A = 0
    possession_B = 0
    assigned_frames = 0  # Count frames where possession is assigned
    last_possession = None  # Track last team with possession ("A" for team A, "B" for team B)

    # Iterate over each timestamp in the dataset
    for _, row in df.iterrows():
        # Get ball coordinates at the current time sample
        ball_pos = np.array([row["ball_x"] * 105/2, row["ball_y"] * 68/2])
        
        # Calculate Euclidean distances for team A players
        distances_A = [
            np.linalg.norm(ball_pos - np.array([row[col_x] * 105/2, row[col_y] * 68/2]))
            for col_x, col_y in team_A
        ]
        min_distance_A = min(distances_A)
        
        # Calculate Euclidean distances for team B players
        distances_B = [
            np.linalg.norm(ball_pos - np.array([row[col_x] * 105/2, row[col_y] * 68/2]))
            for col_x, col_y in team_B
        ]
        min_distance_B = min(distances_B)

        # If both teams are too far from the ball...
        if min_distance_A > threshold and min_distance_B > threshold:
            # ... assign possession to the last team that had it, if available.
            if last_possession is not None:
                assigned_frames += 1
                if last_possession == "A":
                    possession_A += 1
                elif last_possession == "B":
                    possession_B += 1
            # If no team previously had possession, skip this frame.
            continue
        
        # Otherwise, assign possession to the team with the closest player.
        assigned_frames += 1
        if min_distance_A < min_distance_B:
            possession_A += 1
            last_possession = "A"
        else:
            possession_B += 1
            last_possession = "B"

    # Avoid division by zero in case no frames were assigned.
    if assigned_frames == 0:
        return 0.0, 0.0

    # Calculate possession percentage for each team.
    percentage_A = (possession_A / assigned_frames) * 100
    percentage_B = (possession_B / assigned_frames) * 100

    return percentage_A, percentage_B


if __name__ == "__main__":
    csv_file = "/home/tzikos/Desktop/jsons/train/stitched_game_0.csv"  # Replace with your actual CSV file path
    team_A_possession, team_B_possession = compute_possession(csv_file)
    print("Home Team Possession: {:.2f}%".format(team_A_possession))
    print("Away Team Possession: {:.2f}%".format(team_B_possession))
