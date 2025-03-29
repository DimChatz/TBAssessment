import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import SlidingWindowTrajectoryDataset
from models import MLSTMFCNRegression
from train_eval_test import EarlyStopping, trainer, inferencer, validator   
import pandas as pd

def main():
    # Config
    train_path = "/home/tzikos/Desktop/jsons/train"
    val_path = "/home/tzikos/Desktop/jsons/val"
    test_path = "/home/tzikos/Desktop/jsons/inference"
    window_size = 150
    step = 1
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-3
    patience = 13
    max_epochs = 1000
    save_path = "best_model.pt"
    TRAIN = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = SlidingWindowTrajectoryDataset(train_path, window_size, step, inference=False, use_ball=True)
    val_dataset = SlidingWindowTrajectoryDataset(val_path, window_size, step, inference=False, use_ball=True)
    test_dataset = SlidingWindowTrajectoryDataset(test_path, window_size, step, inference=True, use_ball=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model, loss, optimizer
    model = MLSTMFCNRegression().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.1, verbose=True)

    early_stopper = EarlyStopping(patience=patience, save_path=save_path)
    if TRAIN:
        # Training loop
        for epoch in range(max_epochs):
            train_mse = trainer(model, train_loader, optimizer, criterion, device)
            val_mse = validator(model, val_loader, criterion, device)

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
    predictions = inferencer(model, test_loader, device)
    print("Inference complete. Predictions shape:", predictions.shape)

    # Save to CSV
    df = pd.DataFrame(predictions)
    df.to_csv("inference_results.csv", index=False)
    print("Predictions saved to inference_results.csv")


if __name__ == "__main__":
    main()
