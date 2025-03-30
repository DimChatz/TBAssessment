import pandas as pd
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from typing import Optional

# Define pitch dimensions (in normalized units)
PITCH_LENGTH: float = 1.0
PITCH_WIDTH: float = 1.0

def draw_frame_from_row(
    row: pd.Series,
    pred_ball_pos: Optional[pd.Series] = None,
    ax: plt.Axes = None,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH
) -> None:
    """
    Draws a single frame of a pitch with player and ball positions.

    This function clears the provided matplotlib axis and sets the pitch dimensions. It plots the positions
    of home players (blue), away players (red), the actual ball (black), and an optionally predicted ball
    position (yellow) on the pitch.

    Args:
        row (pd.Series): A row from the game DataFrame containing normalized positions for players and ball.
        pred_ball_pos (Optional[pd.Series]): A row containing predicted ball positions [x, y] in normalized units.
            Defaults to None.
        ax (plt.Axes): The matplotlib axes object on which to draw the frame.
        pitch_length (float): The length of the pitch (normalized units). Defaults to PITCH_LENGTH.
        pitch_width (float): The width of the pitch (normalized units). Defaults to PITCH_WIDTH.

    Returns:
        None: This function does not return a value.
    """
    ax.clear()
    ax.set_xlim(-1, pitch_length)
    ax.set_ylim(-1, pitch_width)
    ax.set_facecolor("green")

    # Plot all home players (blue)
    for i in range(11):  # assuming 11 players
        x = row.get(f'home_{i}_x')
        y = row.get(f'home_{i}_y')
        if pd.notna(x) and pd.notna(y):
            # Multiply normalized values by pitch dimensions
            ax.plot(x * pitch_length, y * pitch_width, 'bo')

    # Plot all away players (red)
    for i in range(11):  # assuming 11 players
        x = row.get(f'away_{i}_x')
        y = row.get(f'away_{i}_y')
        if pd.notna(x) and pd.notna(y):
            ax.plot(x * pitch_length, y * pitch_width, 'ro')

    # Plot actual ball if available (black)
    x = row.get('ball_x')
    y = row.get('ball_y')
    if pd.notna(x) and pd.notna(y):
        ax.plot(x * pitch_length, y * pitch_width, 'ko')

    # Plot predicted ball (yellow)
    if pred_ball_pos is not None:
        # Assuming pred_ball_pos contains normalized [x, y]
        ax.plot(pred_ball_pos.iloc[0] * pitch_length, pred_ball_pos.iloc[1] * pitch_width, 'yo')

def create_video_wide_format(
    game_csv_path: str,
    pred_csv_path: str,
    output_path: str = 'output.mp4',
    fps: int = 10,
    window_size: int = 150
) -> None:
    """
    Creates a video by drawing frames from game data and predicted ball positions.

    This function reads game data from a CSV file and predicted ball positions from another CSV file.
    It generates frames by drawing each game frame with the corresponding predicted ball position (if available),
    saves the frames temporarily, compiles them into a video using imageio, and cleans up the temporary frames.

    Args:
        game_csv_path (str): The file path to the CSV file containing game data with player and ball positions.
        pred_csv_path (str): The file path to the CSV file containing predicted ball positions.
        output_path (str, optional): The output file path for the generated video. Defaults to 'output.mp4'.
        fps (int, optional): Frames per second for the output video. Defaults to 10.
        window_size (int, optional): The sliding window size used to align predictions with game frames. Defaults to 150.

    Returns:
        None: This function does not return a value.
    """
    df_game: pd.DataFrame = pd.read_csv(game_csv_path)
    df_pred: pd.DataFrame = pd.read_csv(pred_csv_path)

    if len(df_game) < len(df_pred):
        raise ValueError("More predictions than frames - check input.")

    expected_preds: int = len(df_game) - window_size + 1
    if len(df_pred) != expected_preds:
        print(f"⚠️ Warning: Expected {expected_preds} predictions but got {len(df_pred)}")

    os.makedirs("frames_tmp", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    frame_paths: list[str] = []

    for i in range(len(df_game)):
        row: pd.Series = df_game.iloc[i]
        pred_ball_pos: Optional[pd.Series] = None
        # Align predictions so that prediction 0 corresponds to frame window_size - 1
        if i >= window_size - 1 and (i - (window_size - 1)) < len(df_pred):
            pred_ball_pos = df_pred.iloc[i - (window_size - 1)]
        draw_frame_from_row(row, pred_ball_pos=pred_ball_pos, ax=ax)
        path: str = f"frames_tmp/frame_{i:04d}.png"
        plt.savefig(path, bbox_inches='tight')
        frame_paths.append(path)

    # Use imageio to write the video
    with imageio.get_writer(output_path, fps=fps, plugin='ffmpeg') as writer:
        for path in frame_paths:
            img = imageio.imread(path)
            writer.append_data(img)

    # Cleanup temporary frame images
    for path in frame_paths:
        os.remove(path)
    os.rmdir("frames_tmp")

if __name__ == "__main__":
    create_video_wide_format(
        "/home/tzikos/Desktop/jsons/inference/stitched_game_0.csv",
        "inference_results.csv",
        output_path="game0_video.mp4",
        fps=60,
        window_size=150
    )
