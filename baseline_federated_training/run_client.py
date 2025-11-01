import argparse
import random
import numpy as np
import torch
import flwr as fl
from pathlib import Path

from model import CustomFashionModel
from client import CustomClient
from utils import load_client_data


def main():
    parser = argparse.ArgumentParser(description="Run a single federated learning client.")
    parser.add_argument("--cid", type=int, required=True, help="Client ID to load data for")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str((Path(__file__).resolve().parent / "distributed_data")),
        help="Path to the directory that stores partitioned client data"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_client_data(args.cid, args.data_dir)

    model = CustomFashionModel().to(device)
    client = CustomClient(model, train_loader, test_loader, device)

    fl.client.start_client(server_address="localhost:8081", client=client)


if __name__ == "__main__":
    main()
