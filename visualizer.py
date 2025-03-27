import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Circle

def visualize_trajectory_video(home_players, away_players, ball_df, output_path="trajectory_video.mp4", field_size=(105, 68), fps=30):
    """
    Create a video visualizing the player and ball trajectories on a field.

    Args:
        home_players (list of DataFrames): Each DataFrame is (T x 2), one per home player
        away_players (list of DataFrames): Each DataFrame is (T x 2), one per away player
        ball_df (DataFrame): Ball trajectory (T x 2)
        output_path (str): Path to output video file (e.g. .mp4)
        field_size (tuple): (length, width) in meters
        fps (int): Frames per second for video
    """
    T = len(ball_df)
    num_home = len(home_players)
    num_away = len(away_players)

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.axis('off')

    temp_img_path = "_frame_temp.png"
    frames = []

    for t in range(T):
        ax.clear()

        # Draw pitch
        length, width = field_size
        ax.set_xlim(0, length)
        ax.set_ylim(0, width)
        ax.set_facecolor('green')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Time Step {t}", color='white')

        # Draw center line and circle
        ax.plot([length/2, length/2], [0, width], color='white', linewidth=2)
        center_circle = plt.Circle((length/2, width/2), 9.15, color='white', fill=False, linewidth=2)
        ax.add_patch(center_circle)

        # Plot players
        for i, player_df in enumerate(home_players):
            if t < len(player_df) and not player_df.iloc[t].isnull().any():
                x, y = player_df.iloc[t]
                ax.plot(x, y, 'o', color='blue', markersize=10)

        for i, player_df in enumerate(away_players):
            if t < len(player_df) and not player_df.iloc[t].isnull().any():
                x, y = player_df.iloc[t]
                ax.plot(x, y, 'o', color='red', markersize=10)

        # Plot ball
        if not ball_df.iloc[t].isnull().any():
            ball_x, ball_y = ball_df.iloc[t]
            ax.plot(ball_x, ball_y, 'o', color='white', markersize=6)

        # Save current frame as image
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        img = cv2.imread(temp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    # Clean up temp image
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    # Write video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    print(f"Video saved to {output_path}")
