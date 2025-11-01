import subprocess
import time
import random

NUM_CLIENTS = 10
MALICIOUS_RATIO = 0.25  # 0, 0.25, or 0.5
ATTACK_TYPE = "data"    # "data" or "model"

random.seed(42) 

# Randomly select malicious clients
num_malicious = int(NUM_CLIENTS * MALICIOUS_RATIO)
malicious_clients = set(random.sample(range(NUM_CLIENTS), num_malicious))

print(f"Malicious clients: {sorted(malicious_clients)}")

processes = []

for cid in range(NUM_CLIENTS):
    if cid in malicious_clients:
        attack_type = ATTACK_TYPE
    else:
        attack_type = "none"
    p = subprocess.Popen(
        ["python", "run_client_attack.py", attack_type, str(cid)]
    )
    processes.append(p)
    time.sleep(1) 

print(f"Launched {NUM_CLIENTS} clients ({num_malicious} malicious, attack type: {ATTACK_TYPE}).")

for p in processes:
    p.wait()
