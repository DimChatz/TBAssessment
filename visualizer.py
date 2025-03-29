import pandas as pd
import matplotlib.pyplot as plt
import os
import imageio

# Define pitch dimensions (in meters, for example)
PITCH_LENGTH = 1
PITCH_WIDTH = 1

def draw_frame_from_row(row, pred_ball_pos=None, ax=None, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
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
        ax.plot(pred_ball_pos[0] * pitch_length, pred_ball_pos[1] * pitch_width, 'yo')

def create_video_wide_format(game_csv_path, pred_csv_path, output_path='output.mp4', fps=10, window_size=150):
    df_game = pd.read_csv(game_csv_path)
    df_pred = pd.read_csv(pred_csv_path)

    if len(df_game) < len(df_pred):
        raise ValueError("More predictions than frames - check input.")

    expected_preds = len(df_game) - window_size + 1
    if len(df_pred) != expected_preds:
        print(f"⚠️ Warning: Expected {expected_preds} predictions but got {len(df_pred)}")

    os.makedirs("frames_tmp", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    frame_paths = []

    for i in range(len(df_game)):
        row = df_game.iloc[i]
        pred_ball_pos = None
        # Align predictions so that prediction 0 corresponds to frame window_size - 1
        if i >= window_size - 1 and (i - (window_size - 1)) < len(df_pred):
            pred_ball_pos = df_pred.iloc[i - (window_size - 1)]
        draw_frame_from_row(row, pred_ball_pos=pred_ball_pos, ax=ax)
        path = f"frames_tmp/frame_{i:04d}.png"
        plt.savefig(path, bbox_inches='tight')
        frame_paths.append(path)

    # Use imageio to write the video
    writer = imageio.get_writer(output_path, fps=fps)
    for path in frame_paths:
        img = imageio.imread(path)
        writer.append_data(img)
    writer.close()

    # Cleanup temporary frame images
    for path in frame_paths:
        os.remove(path)
    os.rmdir("frames_tmp")

if __name__ == "__main__":
    create_video_wide_format(
        "/home/tzikos/Desktop/jsons/inference/stitched_game_2.csv",
        "inference_results.csv",
        output_path="game2_video.mp4",
        fps=60,
        window_size=150
    )
