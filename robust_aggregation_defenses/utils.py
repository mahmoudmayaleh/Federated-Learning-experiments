import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def load_client_data(cid: int, data_dir: str, batch_size: int = 32):
    filepath = os.path.join(data_dir, f"client_{cid}.npz")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Client data not found: {filepath}")

    data = np.load(filepath)
    x = data["x"]
    y = data["y"]

    x = x.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
