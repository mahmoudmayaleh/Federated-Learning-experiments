from utils import load_client_data

train_loader, val_loader = load_client_data(0, "./distributed_data")
print(f"✅ Loaded client 0: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")