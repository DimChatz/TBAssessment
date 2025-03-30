import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Any, Union


def trainer(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, criterion: Any, device: torch.device) -> float:
    """
    Trains the model for one epoch using the provided dataloader.

    This function sets the model to training mode and iterates over the dataloader,
    computing the loss, performing backpropagation, and updating the model parameters.
    The average loss for the epoch is then returned.

    Args:
        model (nn.Module): The neural network model to be trained.
        dataloader (DataLoader): DataLoader providing the training data batches.
        optimizer (Optimizer): Optimizer used to update model parameters.
        criterion (Any): Loss function used to compute the training loss.
        device (torch.device): The device (CPU or GPU) on which computations will be performed.

    Returns:
        float: The average training loss over the epoch.
    """
    model.train()
    total_loss: float = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)   # (B, W, 66)
        targets = targets.to(device) # (B, W, 2)

        optimizer.zero_grad()
        outputs = model(inputs)      # (B, W, 2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss: float = total_loss / len(dataloader)
    return avg_loss


def validator(model: nn.Module, dataloader: DataLoader, criterion: Any, device: torch.device) -> float:
    """
    Evaluates the model on a validation dataset.

    This function sets the model to evaluation mode, computes the loss for each batch in the
    validation dataloader without performing backpropagation, and returns the average loss.

    Args:
        model (nn.Module): The neural network model to be evaluated.
        dataloader (DataLoader): DataLoader providing the validation data batches.
        criterion (Any): Loss function used to compute the validation loss.
        device (torch.device): The device (CPU or GPU) on which computations will be performed.

    Returns:
        float: The average validation loss over the dataset.
    """
    model.eval()
    total_loss: float = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss: float = total_loss / len(dataloader)
    return avg_loss


def inferencer(model: nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Performs inference on the provided dataloader and returns predictions.

    The function sets the model to evaluation mode and iterates over the dataloader,
    collecting model outputs (predictions) for each input batch. If the dataloader yields
    a tuple (inputs, targets), only the inputs are used for inference.

    Args:
        model (nn.Module): The neural network model used for inference.
        dataloader (DataLoader): DataLoader providing the inference data.
        device (torch.device): The device (CPU or GPU) on which computations will be performed.

    Returns:
        torch.Tensor: A tensor containing all predictions concatenated along the batch dimension.
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for inputs in dataloader:
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]  # Remove dummy targets if present

            inputs = inputs.to(device)
            outputs = model(inputs)  # (B, W, 2)
            all_predictions.append(outputs.cpu())

    return torch.cat(all_predictions, dim=0)  # (total_windows, W, 2)


class EarlyStopping:
    """
    Implements early stopping for training to prevent overfitting.

    The early stopping mechanism monitors a specified validation metric and stops training
    if the metric does not improve beyond a given delta for a specified number of epochs (patience).
    When an improvement is observed, the model state is saved.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
        save_path (str): File path to save the best model state.
        best_metric (float): The best recorded validation metric (lower is better).
        counter (int): Number of consecutive epochs without improvement.
        early_stop (bool): Flag indicating whether early stopping has been triggered.
    """

    def __init__(self, patience: int = 5, delta: float = 0.0, save_path: str = "checkpoint.pt") -> None:
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 5.
            delta (float, optional): Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.0.
            save_path (str, optional): File path to save the best model state. Defaults to "checkpoint.pt".
        """
        self.patience: int = patience
        self.delta: float = delta
        self.save_path: str = save_path
        self.best_metric: float = float("inf")  # Lower is better
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, val_metric: float, model: nn.Module) -> None:
        """
        Checks the validation metric and updates early stopping status.

        If the current validation metric improves (i.e., decreases by more than delta), the best_metric
        is updated, the counter is reset, and the model state is saved. Otherwise, the counter is incremented.
        If the counter exceeds the patience, early stopping is triggered.

        Args:
            val_metric (float): The current validation metric.
            model (nn.Module): The model to be saved if an improvement is observed.

        Returns:
            None
        """
        if val_metric < self.best_metric - self.delta:
            self.best_metric = val_metric
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True