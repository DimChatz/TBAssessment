import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset


class SlidingWindowTrajectoryDataset(Dataset):
    def __init__(self, folder_path, window_size=200, step=1, inference=False, use_ball=True):
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

            # Drop rows with NaNs in the first 66 columns (input features)
            initial_len = len(df)
            df = df.dropna(subset=df.columns[:66])
            if len(df) < initial_len:
                print(f"⚠️ Dropped {initial_len - len(df)} rows with NaNs in: {file_path}")

            if df.shape[0] < self.window_size:
                print(f"Skipping {file_path} — not enough rows for window")
                continue

            input_data = df.iloc[:, :66].values
            input_data = (input_data - input_data.mean(axis=0)) / (input_data.std(axis=0) + 1e-6)

            target_data = df.iloc[:, -2:].values
            for start in range(0, len(df) - self.window_size + 1, self.step):
                end = start + self.window_size
                input_window = torch.tensor(input_data[start:end], dtype=torch.float32).T

                self.inputs.append(input_window)
                if not self.inference and self.use_ball:
                    self.targets.append(torch.tensor(target_data[start:end], dtype=torch.float32)[-1])
                elif not self.inference:
                    self.targets.append(torch.empty(self.window_size, 0))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.inference or not self.use_ball:
            return self.inputs[idx]
        return self.inputs[idx], self.targets[idx]
