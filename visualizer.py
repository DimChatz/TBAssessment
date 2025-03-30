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
        ax.plot(pred_ball_pos.iloc[0] * pitch_length, pred_ball_pos.iloc[1] * pitch_width, 'yo')

def create_gif_from_frames(
    game_csv_path: str,
    pred_csv_path: str,
    output_path: str = 'output.gif',
    fps: int = 10,
    window_size: int = 150,
    start_frame: int = 200,
    end_frame: int = 1000
) -> None:
    """
    Creates a GIF by drawing frames from game data and predicted ball positions.
    Only frames between `start_frame` and `end_frame` (from the original game data) are processed.
    """
    # Read the full game CSV (we are now selecting frames by index)
    df_game: pd.DataFrame = pd.read_csv(game_csv_path)
    df_pred: pd.DataFrame = pd.read_csv(pred_csv_path)
    
    # Calculate expected predictions for the given frame range.
    # Since predictions are aligned such that prediction 0 corresponds to frame window_size-1,
    # for frame i we use prediction at index (i - (window_size-1)).
    # For our subset, the first frame (i=start_frame) uses prediction index = start_frame - (window_size-1)
    pred_start_index = start_frame - (window_size - 1)
    pred_end_index = end_frame - (window_size - 1)  # exclusive end for predictions
    expected_preds = end_frame - start_frame  # Number of frames we expect to have a prediction

    if len(df_pred) > pred_end_index:
        print(f"⚠️ Warning: More predictions ({len(df_pred)}) than expected up to index {pred_end_index}.")
    elif len(df_pred) < pred_end_index:
        print(f"⚠️ Warning: Fewer predictions than expected: expected at least {pred_end_index} but got {len(df_pred)}.")
    
    os.makedirs("frames_tmp", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    frame_paths: list[str] = []
    frame_counter = 0

    # Iterate over original frame indices between start_frame and end_frame.
    for i in range(start_frame, end_frame):
        row: pd.Series = df_game.iloc[i]
        pred_ball_pos: Optional[pd.Series] = None
        # Since the prediction for frame i is at index i - (window_size - 1), we check bounds.
        pred_idx = i - (window_size - 1)
        if pred_idx >= 0 and pred_idx < len(df_pred):
            pred_ball_pos = df_pred.iloc[pred_idx]
        draw_frame_from_row(row, pred_ball_pos=pred_ball_pos, ax=ax)
        path: str = f"frames_tmp/frame_{frame_counter:04d}.png"
        plt.savefig(path, bbox_inches='tight')
        frame_paths.append(path)
        frame_counter += 1

    # Calculate duration per frame for GIF (in seconds)
    duration_per_frame = 1 / fps
    with imageio.get_writer(output_path, mode='I', duration=duration_per_frame) as writer:
        for path in frame_paths:
            img = imageio.imread(path)
            writer.append_data(img)

    # Cleanup temporary frame images
    for path in frame_paths:
        os.remove(path)
    os.rmdir("frames_tmp")

if __name__ == "__main__":
    create_gif_from_frames(
        "/home/tzikos/Desktop/jsons/inference/stitched_game_4.csv",
        "inference_results.csv",
        output_path="game0_video.gif",
        fps=60,
        window_size=150,
        start_frame=200,
        end_frame=1000  # Only frames from the 200th to 1000th are processed
    )
