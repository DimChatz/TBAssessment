import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset


class SlidingWindowTrajectoryDataset(Dataset):
    """Dataset for sliding window trajectory extraction from CSV files.

    This dataset reads CSV files from a specified folder and processes them into sliding windows.
    Each window is normalized based on the input features and targets are extracted from the last two
    columns if applicable.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        window_size (int, optional): Number of rows per sliding window. Defaults to 200.
        step (int, optional): Step size between consecutive windows. Defaults to 1.
        inference (bool, optional): If True, only inputs are returned without targets. Defaults to False.
        use_ball (bool, optional): If True and not in inference mode, the target is extracted from the input.
                                    Otherwise, an empty tensor is used as the target. Defaults to True.
    """
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

            # Skip file if not enough rows for one window
            if df.shape[0] < self.window_size:
                print(f"Skipping {file_path} — not enough rows for window")
                continue

            # Normalize the input features (first 66 columns)
            input_data = df.iloc[:, :66].values
            input_data = (input_data - input_data.mean(axis=0)) / (input_data.std(axis=0) + 1e-6)

            # Extract target data from the last 2 columns
            target_data = df.iloc[:, -2:].values

            # Generate sliding windows
            for start in range(0, len(df) - self.window_size + 1, self.step):
                end = start + self.window_size
                # Transpose so that shape becomes (channels, window_size)
                input_window = torch.tensor(input_data[start:end], dtype=torch.float32).T

                self.inputs.append(input_window)
                if not self.inference and self.use_ball:
                    # Use the target of the last row in the window
                    self.targets.append(torch.tensor(target_data[start:end], dtype=torch.float32)[-1])
                elif not self.inference:
                    # If use_ball is False, create an empty target tensor with shape (window_size, 0)
                    self.targets.append(torch.empty(self.window_size, 0))

    def __len__(self):
        """Returns the number of sliding windows in the dataset.

        Returns:
            int: Total number of windows.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """Retrieves the input window (and target if applicable) at the specified index.

        Args:
            idx (int): Index of the data item.

        Returns:
            torch.Tensor or tuple: If in inference mode or if use_ball is False, returns the input window.
            Otherwise, returns a tuple (input_window, target).
        """
        if self.inference or not self.use_ball:
            return self.inputs[idx]
        return self.inputs[idx], self.targets[idx]
