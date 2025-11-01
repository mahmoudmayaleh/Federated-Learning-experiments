import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from torch.utils.data import DataLoader

class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device, mu=0.0, global_params=None) -> tuple[float, float]:
        self.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            if mu > 0 and global_params is not None:
                prox = 0.0
                for w, w0 in zip(self.parameters(), global_params):
                    prox += ((w - w0).norm(2) ** 2)
                loss += (mu / 2) * prox
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def test_epoch(self, test_loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> tuple[float, float]:
        self.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return loss / total, correct / total

    def get_model_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.state_dict().values()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.state_dict()
        for key, param in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(param)
        self.load_state_dict(state_dict, strict=True)
