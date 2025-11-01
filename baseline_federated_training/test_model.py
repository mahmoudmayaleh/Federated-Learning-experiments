import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import CustomFashionModel

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy data: 100 samples, 1 channel, 28x28 (like FashionMNIST)
    inputs = torch.randn(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))

    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=10)

    model = CustomFashionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Forward pass test
    model.eval()
    with torch.no_grad():
        sample_output = model(inputs.to(device))
    print("Forward pass output shape:", sample_output.shape)  # Should be [100, 10]

    # Train epoch test
    loss, acc = model.train_epoch(loader, criterion, optimizer, device)
    print(f"Train epoch -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Test epoch test
    loss, acc = model.test_epoch(loader, criterion, device)
    print(f"Test epoch -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    test_model()
