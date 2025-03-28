import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset


class SlidingWindowTrajectoryDataset(Dataset):
    def __init__(self, folder_path, window_size=200, step=1, inference=False, use_ball=True):
        """
        Args:
            folder_path (str): Path to folder containing CSV files.
            window_size (int): Number of timesteps in each window.
            step (int): Step size between windows.
            inference (bool): If True, ignore target (used for inference).
            use_ball (bool): If False, ignore last 2 columns even during training/validation.
        """
        self.window_size = window_size
        self.step = step
        self.inference = inference
        self.use_ball = use_ball

        self.inputs = []
        self.targets = []

        # Collect and sort CSV files
        file_list = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

        for file_path in file_list:
            df = pd.read_csv(file_path)

            # Extract input and target
            input_data = df.iloc[:, :66].values
            target_data = df.iloc[:, -2:].values  # last 2 columns assumed to be ball position

            num_frames = len(df)
            for start in range(0, num_frames - window_size + 1, step):
                end = start + window_size

                input_window = torch.tensor(input_data[start:end], dtype=torch.float32)  # (window_size, 66)
                self.inputs.append(input_window)

                if not inference and use_ball:
                    target_window = torch.tensor(target_data[start:end], dtype=torch.float32)  # (window_size, 2)
                    self.targets.append(target_window)
                elif not inference and not use_ball:
                    self.targets.append(torch.empty(window_size, 0))  # Placeholder

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.inference or not self.use_ball:
            return self.inputs[idx]
        return self.inputs[idx], self.targets[idx]
