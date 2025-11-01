import subprocess
import time

NUM_CLIENTS = 10

processes = []

for cid in range(NUM_CLIENTS):
    p = subprocess.Popen(
        ["python", "run_client.py", "--cid", str(cid)]
    )
    processes.append(p)
    time.sleep(1)

print(f"Launched {NUM_CLIENTS} clients.")

for p in processes:
    p.wait()
