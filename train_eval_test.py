import torch

def trainer(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)                # (B, W, 66)
        targets = targets.to(device)              # (B, W, 2)

        optimizer.zero_grad()
        outputs = model(inputs)                   # (B, W, 2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def validator(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def inferencer(model, dataloader, device):
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
    def __init__(self, patience=5, delta=0.0, save_path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True