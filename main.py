import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import SlidingWindowTrajectoryDataset
from models import MLSTMFCNRegression
from train_eval_test import EarlyStopping, trainer, inferencer, validator
import pandas as pd
from typing import Any

def main() -> None:
    """
    Main function to train, validate, and perform inference with the MLSTMFCNRegression model.

    This function sets the configuration for training and inference, loads the training, validation,
    and test datasets using a sliding window approach, initializes the model, loss function, optimizer,
    and learning rate scheduler, and optionally trains the model with early stopping. After training,
    the best model is loaded, inference is performed on the test dataset, and the predictions are saved to a CSV file.

    Returns:
        None
    """
    # Configurations
    train_path: str = "/home/tzikos/Desktop/jsons/train"
    val_path: str = "/home/tzikos/Desktop/jsons/val"
    test_path: str = "/home/tzikos/Desktop/jsons/inference"
    window_size: int = 150
    step: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    patience: int = 13
    max_epochs: int = 1000
    save_path: str = "best_model.pt"
    TRAIN: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = SlidingWindowTrajectoryDataset(train_path, window_size, step, inference=False, use_ball=True)
    val_dataset = SlidingWindowTrajectoryDataset(val_path, window_size, step, inference=False, use_ball=True)
    test_dataset = SlidingWindowTrajectoryDataset(test_path, window_size, step, inference=True, use_ball=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model, loss, optimizer, and scheduler setup
    model: MLSTMFCNRegression = MLSTMFCNRegression().to(device)
    criterion: nn.Module = nn.MSELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.1, verbose=True)

    early_stopper = EarlyStopping(patience=patience, save_path=save_path)
    
    if TRAIN:
        # Training loop
        for epoch in range(max_epochs):
            train_mse: float = trainer(model, train_loader, optimizer, criterion, device)
            val_mse: float = validator(model, val_loader, criterion, device)

            scheduler.step(val_mse)
            early_stopper(val_mse, model)

            print(f"Epoch {epoch+1}/{max_epochs}")
            print(f"  Training MSE:   {train_mse:.6f}")
            print(f"  Validation MSE: {val_mse:.6f}")

            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load("/home/tzikos/Desktop/TBAssessment/B8_LR4_L23_MSE239_fold4.pt"))

    # Inference
    predictions: Any = inferencer(model, test_loader, device)
    print("Inference complete. Predictions shape:", predictions.shape)

    # Save predictions to CSV
    df: pd.DataFrame = pd.DataFrame(predictions)
    df.to_csv("inference_results.csv", index=False)
    print("Predictions saved to inference_results.csv")


if __name__ == "__main__":
    main()