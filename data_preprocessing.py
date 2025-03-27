import pandas as pd
import re
import argparse
import os


def load_and_stitch_players(excel_path: str, use_ball: bool = True) -> dict:
    # Step 1: Read Excel or CSV
    try:
        df = pd.read_excel(excel_path)
    except:
        df = pd.read_csv(excel_path)

    # Step 2: Identify player columns (exclude ball)
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

    # Step 4: Stitching logic
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
            new_contribution = (~nonnull_q & ~nonnull_p).sum()

            if overlap > 3 or new_contribution > 3:
                continue

            stitched_df.loc[nonnull_q] = other_data.loc[nonnull_q]
            assigned.add(q)

        stitched_players.append(stitched_df)

    # Step 5: Extract ball if required
    if use_ball:
        ball_cols = [col for col in df.columns if re.match(r'^ball_[xy]$', col)]
        ball_df = df[ball_cols].copy()
    else:
        ball_df = pd.DataFrame(index=df.index)

    return {
        'players': stitched_players,
        'ball': ball_df
    }


def main():
    parser = argparse.ArgumentParser(description="Stitch player trajectories from Excel or CSV.")
    parser.add_argument("--input_dir", required=True, help="Directory containing 'Home' and 'Away' files")
    parser.add_argument("--output_dir", required=True, help="Directory to save stitched CSV")
    parser.add_argument("--data_format", default=True, help="True for CSV, False for Excel")
    parser.add_argument("--use_ball", action="store_true", help="Include ball data if set")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    final_df = pd.DataFrame()

    for team, prefix, z_value in [('Home', 'home', 1), ('Away', 'away', -1)]:
        path_type = "csv" if args.data_format else "xlsx"
        input_path = os.path.join(args.input_dir, f"{team}.{path_type}")
        result = load_and_stitch_players(input_path, use_ball=args.use_ball)

        players = result['players'][:11]

        for i, player_df in enumerate(players):
            renamed = player_df.copy()
            renamed.columns = [f"{prefix}_{i}_x", f"{prefix}_{i}_y"]
            z_col = pd.Series(z_value, index=renamed.index, name=f"{prefix}_{i}_z")
            final_df = pd.concat([final_df, renamed, z_col], axis=1)

    # Add ball if needed
    if args.use_ball and not result['ball'].empty:
        ball_df = result['ball'].copy()
        ball_df.columns = ['ball_x', 'ball_y']
        final_df = pd.concat([final_df, ball_df], axis=1)

    # Forward-fill missing values
    final_df = final_df.ffill()

    # Save to CSV
    output_path = os.path.join(args.output_dir, "stitched_game.csv")
    print(f"Saved to: {output_path}")
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
