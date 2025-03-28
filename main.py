import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import SlidingWindowTrajectoryDataset
from models import MLSTMFCN  # Replace with your model
from train_eval_test import EarlyStopping, trainer, inferencer, validator   


def main():
    # Config
    train_path = "./data/train"
    val_path = "./data/val"
    test_path = "./data/test"
    window_size = 200
    step = 20
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-4  # L2 regularization
    patience = 13
    max_epochs = 1000
    save_path = "best_model.pt"

    device = torch.device("cpu")

    # Load datasets
    train_dataset = SlidingWindowTrajectoryDataset(train_path, window_size, step, inference=False, use_ball=True)
    val_dataset = SlidingWindowTrajectoryDataset(val_path, window_size, step, inference=False, use_ball=True)
    test_dataset = SlidingWindowTrajectoryDataset(test_path, window_size, step, inference=True, use_ball=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model, loss, optimizer
    model = MLSTMFCN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.1, verbose=True)

    early_stopper = EarlyStopping(patience=patience, save_path=save_path)

    # Training loop
    for epoch in range(max_epochs):
        train_loss = trainer(model, train_loader, optimizer, criterion, device)
        val_loss = validator(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        early_stopper(val_loss, model)

        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(torch.load(save_path))

    # Inference
    predictions = inferencer(model, test_loader, device)
    print("Inference complete. Predictions shape:", predictions.shape)


if __name__ == "__main__":
    main()