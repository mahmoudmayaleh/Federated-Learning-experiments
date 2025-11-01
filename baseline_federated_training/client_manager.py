import time
from typing import Dict, List, Optional
from threading import Lock, Condition
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

class CustomClientManager(ClientManager):
    def __init__(self):
        self.clients: Dict[str, ClientProxy] = {}
        self.lock = Lock()
        self.cond = Condition(self.lock)

    def num_available(self) -> int:
        with self.lock:
            return len(self.clients)

    def register(self, client: ClientProxy) -> bool:
        with self.lock:
            self.clients[client.cid] = client
            self.cond.notify_all()  # Notify waiting threads
        return True

    def unregister(self, client: ClientProxy) -> None:
        with self.lock:
            if client.cid in self.clients:
                del self.clients[client.cid]
                self.cond.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        with self.lock:
            return dict(self.clients)

    def sample(self, num_clients: int,
               min_num_clients: Optional[int] = None,
               criterion: Optional[object] = None) -> List[ClientProxy]:
        with self.cond:
            # Wait until enough clients are available
            while len(self.clients) < num_clients:
                self.cond.wait(timeout=1)  # Wait for 1 second, then check again
            # Now sample clients (simple random sample)
            import random
            return random.sample(list(self.clients.values()), num_clients)

    def wait_for(self, num_clients: int, timeout: Optional[float] = None) -> bool:
        """Wait until at least `num_clients` are registered, or until `timeout` seconds have passed."""
        import time
        start = time.time()
        with self.cond:
            while len(self.clients) < num_clients:
                remaining = None if timeout is None else max(0, timeout - (time.time() - start))
                if remaining == 0:
                    return False
                self.cond.wait(timeout=remaining)
        return True