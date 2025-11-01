import os
import numpy as np
from torchvision import datasets, transforms

def generate_distributed_datasets(k: int, alpha: float, save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    data = trainset.data.numpy()
    targets = trainset.targets.numpy()
    num_classes = 10
    client_data = {i: {"x": [], "y": []} for i in range(k)}

    class_indices = {i: np.where(targets == i)[0] for i in range(num_classes)}
    np.random.seed(42)
    
    for c in range(num_classes):
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, k))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        class_split = np.split(class_indices[c], proportions)
        for i, idx in enumerate(class_split):
            client_data[i]["x"].extend(data[idx])
            client_data[i]["y"].extend(targets[idx])

    for i in range(k):
        x = np.array(client_data[i]["x"])
        y = np.array(client_data[i]["y"])
        np.savez(os.path.join(save_dir, f"client_{i}.npz"), x=x, y=y)

    print(f"✅ Saved distributed datasets in '{save_dir}' for {k} clients (α={alpha})")

# This script generates distributed datasets for federated learning clients
if __name__ == "__main__":
    NUM_CLIENTS = 10
    ALPHA_DIRICHLET = 1.0
    generate_distributed_datasets(
        k=NUM_CLIENTS,
        alpha=ALPHA_DIRICHLET,
        save_dir="distributed_data"
    )
