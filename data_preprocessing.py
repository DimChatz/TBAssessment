import pandas as pd
import re
import argparse
import os
from typing import Dict, Any


def load_and_stitch_players(excel_path: str, use_ball: bool = True) -> Dict[str, Any]:
    """
    Load and stitch player trajectories from an Excel or CSV file.

    This function reads player trajectory data from the provided file, extracts player coordinate columns
    (ignoring ball data if specified), and stitches pairs of columns representing the same player. It then
    optionally extracts ball data.

    Args:
        excel_path (str): The path to the Excel or CSV file containing player data.
        use_ball (bool, optional): Whether to include ball data. If True, ball columns are extracted.
            Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary with the following keys:
            - 'players': A list of pandas DataFrames, each containing stitched player coordinates.
            - 'ball': A pandas DataFrame containing ball coordinate data if use_ball is True,
              otherwise an empty DataFrame.
    """
    # Step 1: Read Excel or CSV
    try:
        df = pd.read_excel(excel_path)
    except Exception:
        df = pd.read_csv(excel_path)

    # Step 2: Identify player columns (exclude ball if needed)
    coord_cols = df.columns[3:-2] if use_ball else df.columns[3:]
    player_data = {}
    i = 0

    # Step 3: Extract players from column pairs
    while i < len(coord_cols) - 1:
        x_col = coord_cols[i]
        y_col = coord_cols[i + 1]
        base_name_x = x_col[:-2] if x_col.endswith('_x') else x_col
        base_name_y = y_col[:-2] if y_col.endswith('_y') else y_col

        if base_name_x != base_name_y:
            print(f"Warning: Mismatched player columns: {x_col}, {y_col}")
            i += 1
            continue

        if x_col in df.columns and y_col in df.columns:
            player_data[base_name_x] = df[[x_col, y_col]].copy()
        else:
            print(f"Warning: Missing columns for {base_name_x}")

        i += 2

    # Step 4: Stitching logic for players
    stitched_players = []
    assigned = set()
    player_names = list(player_data.keys())

    for p in player_names:
        if p in assigned:
            continue

        base_data = player_data[p]
        stitched_df = base_data.copy()
        assigned.add(p)

        for q in player_names:
            if q in assigned or q == p:
                continue

            other_data = player_data[q]
            nonnull_p = base_data.notna().all(axis=1)
            nonnull_q = other_data.notna().all(axis=1)

            overlap = (nonnull_p & nonnull_q).sum()
            new_contribution = ((~nonnull_q) & (~nonnull_p)).sum()

            if overlap > 15 or new_contribution > 15:
                continue

            stitched_df.loc[nonnull_q] = other_data.loc[nonnull_q]
            assigned.add(q)

        stitched_players.append(stitched_df)

    # Step 5: Extract ball data if required
    if use_ball:
        ball_cols = [col for col in df.columns if re.match(r'^ball_[xy]$', col)]
        ball_df = df[ball_cols].copy()
    else:
        ball_df = pd.DataFrame(index=df.index)

    return {
        'players': stitched_players,
        'ball': ball_df
    }


def main() -> None:
    """
    Main function to stitch player trajectories and save the final data to a CSV file.

    This function:
      1. Parses command-line arguments to determine input and output directories, file format, and ball data usage.
      2. Processes player data for 'Home' and 'Away' teams by reading and stitching the corresponding files.
      3. Concatenates the stitched player data (and ball data, if specified), forward-fills missing values,
         and normalizes each column.
      4. Saves the resulting DataFrame to a CSV file in the output directory.
    """
    parser = argparse.ArgumentParser(
        description="Stitch player trajectories from Excel or CSV."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing 'Home' and 'Away' files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save stitched CSV")
    parser.add_argument("--data_format", action="store_true",
                        help="Use CSV format if set; Excel otherwise")
    parser.add_argument("--use_ball", action="store_true",
                        help="Include ball data if set")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    final_df = pd.DataFrame()

    for team, prefix, z_value in [('Home', 'home', 1), ('Away', 'away', -1)]:
        path_type = "csv" if args.data_format else "xlsx"
        input_path = os.path.join(args.input_dir, f"{team}.{path_type}")
        result = load_and_stitch_players(input_path, use_ball=args.use_ball)

        # Use only the first 11 players
        players = result['players'][:11]

        for i, player_df in enumerate(players):
            renamed = player_df.copy()
            renamed.columns = [f"{prefix}_{i}_x", f"{prefix}_{i}_y"]
            z_col = pd.Series(z_value, index=renamed.index, name=f"{prefix}_{i}_z")
            final_df = pd.concat([final_df, renamed, z_col], axis=1)

    # Add ball data if needed
    if args.use_ball and not result['ball'].empty:
        ball_df = result['ball'].copy()
        ball_df.columns = ['ball_x', 'ball_y']
        final_df = pd.concat([final_df, ball_df], axis=1)

    # Forward-fill missing values
    final_df = final_df.ffill()

    # Normalize each column by the maximum of its absolute value
    final_df = final_df.apply(lambda col: col / col.abs().max(), axis=0)

    # Save the final stitched DataFrame to a CSV file
    output_path = os.path.join(args.output_dir, "stitched_game.csv")
    print(f"Saved to: {output_path}")
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
