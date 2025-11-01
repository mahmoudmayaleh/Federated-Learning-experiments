import multiprocessing
import threading
import random
import time

STRESS_DURATION = 5 * 60  # 5 minutes
SLEEP_DURATION = 3 * 60   # 3 minutes


def stress_cpu(duration):
    def burn():
        end = time.time() + duration
        x = 0.0001
        while time.time() < end:
            x = x * x + 0.0001  # Safe CPU-intensive work

    cores = random.randint(1, multiprocessing.cpu_count())
    print(f"[STRESS] CPU: {cores} core(s) for {duration:.1f}s")
    processes = [multiprocessing.Process(target=burn) for _ in range(cores)]
    for p in processes: p.start()
    for p in processes: p.join()

def stress_ram(size_mb, duration):
    print(f"[STRESS] RAM: {size_mb} MB for {duration:.1f}s")
    try:
        block = bytearray(size_mb * 1024 * 1024)
        time.sleep(duration)
        del block
    except MemoryError:
        print("[STRESS] MemoryError: Could not allocate RAM.")

def stress_worker(stop_event):
    while not stop_event.is_set():
        action = random.choice(["cpu", "ram", "both", "sleep"])
        duration = random.uniform(2, 10)

        if stop_event.is_set():
            break

        if action == "sleep":
            print(f"[STRESS] Sleeping for {duration:.1f}s")
            time.sleep(duration)
        elif action == "cpu":
            stress_cpu(duration)
        elif action == "ram":
            size_mb = random.randint(100, 1000)
            stress_ram(size_mb, duration)
        elif action == "both":
            size_mb = random.randint(100, 1000)
            print(f"[STRESS] CPU + RAM for {duration:.1f}s and {size_mb}MB")
            thread1 = threading.Thread(target=stress_cpu, args=(duration,))
            thread2 = threading.Thread(target=stress_ram, args=(size_mb, duration))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()

        time.sleep(random.uniform(1, 3))  # Short cooldown between actions



def start_stress_in_background():
    stop_event = threading.Event()
    thread = threading.Thread(target=stress_worker, args=(stop_event,))
    thread.daemon = True
    thread.start()
    return stop_event, thread

