import torch
from torch.utils.data import Dataset

class TrajectoryWindowDataset(Dataset):
    def __init__(self, trajectory_dict, window_size=200, step=1):
        """
        Initializes the dataset with sliding windows from the trajectory dict.

        Args:
            trajectory_dict (dict): Contains 'players' (list of DataFrames) and 'ball' (DataFrame)
            window_size (int): Number of timesteps in each window
            step (int): Step size for sliding window
        """
        self.window_size = window_size
        self.inputs = []
        self.targets = []

        # Number of total timesteps
        sequence_length = len(trajectory_dict['ball'])

        # Create sliding windows
        for start in range(0, sequence_length - window_size + 1, step):
            end = start + window_size

            # Stack all players into tensor of shape (num_players, window_size, 2)
            player_window = [
                torch.tensor(player_df.iloc[start:end].values, dtype=torch.float32)
                for player_df in trajectory_dict['players']
            ]
            player_tensor = torch.stack(player_window)  # (num_players, window_size, 2)

            # Ball tensor: (window_size, 2)
            ball_tensor = torch.tensor(
                trajectory_dict['ball'].iloc[start:end].values,
                dtype=torch.float32
            )

            self.inputs.append(player_tensor)
            self.targets.append(ball_tensor)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
