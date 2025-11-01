import sys
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import CustomFashionModel
from client_attack import FedAvgAttackClient
import numpy as np
def load_data(batch_size=64, alpha=1, num_clients=10, seed=42):
    # Download FashionMNIST and split among clients using Dirichlet distribution
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    # Dirichlet split for non-IID
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    client_indices = [[] for _ in range(num_clients)]
    for k in range(10):  # 10 classes
        idx_k = idxs[labels == k]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split_idx = np.split(idx_k, proportions)
        for cid, idx in enumerate(split_idx):
            client_indices[cid].extend(idx)
    # Shuffle indices for each client
    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])

    # Assign client id from environment or argument (default 0)
    client_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    train_subset = torch.utils.data.Subset(dataset, client_indices[client_id])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    batch_size = 64
    alpha = 1
    num_clients = 10
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomFashionModel().to(device)
    train_loader, test_loader = load_data(batch_size, alpha, num_clients, seed)

    attack_type = "none"
    if len(sys.argv) > 1:
        attack_type = sys.argv[1]  # "none", "data", or "model"

    client = FedAvgAttackClient(model, train_loader, test_loader, device, attack_type=attack_type)
    import flwr as fl
    fl.client.start_client(server_address="localhost:8081", client=client.to_client())

if __name__ == "__main__":
    main()
