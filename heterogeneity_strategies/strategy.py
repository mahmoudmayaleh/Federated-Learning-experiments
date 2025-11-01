from typing import Dict, List, Optional, Tuple
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
import numpy as np
import json


class FedAvgStrategy(Strategy):
    def __init__(
        self,
        num_clients: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        alpha_dirichlet: float,
        dataset_name: str,
        seed: int,
    ):
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha_dirichlet = alpha_dirichlet
        self.dataset_name = dataset_name
        self.seed = seed
        self.initial_parameters: Optional[Parameters] = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(self.num_clients)
        config = {} 
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        weights = []
        num_examples = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights.append(ndarrays)
            num_examples.append(fit_res.num_examples)

        if not weights:
            return None, {}

        total = sum(num_examples)
        avg_weights = [
            np.sum([weights[client_idx][i] * num_examples[client_idx] for client_idx in range(len(weights))], axis=0) / total
            for i in range(len(weights[0]))
        ]

        avg_loss = float(np.mean(np.array([fit_res.metrics["FL_loss"] for _, fit_res in results if "FL_loss" in fit_res.metrics])))
        accuracies = [fit_res.metrics["FL_accuracy"] for _, fit_res in results if "FL_accuracy" in fit_res.metrics]
        avg_accuracy = float(np.mean(np.array(accuracies))) if accuracies else 0.0

        return ndarrays_to_parameters(avg_weights), {"FL_loss": avg_loss, "FL_accuracy": avg_accuracy}

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.sample(self.num_clients)
        config = {}
        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        losses = [res.loss for _, res in results]
        accuracies = [res.metrics["FL_accuracy"] for _, res in results if "FL_accuracy" in res.metrics]

        avg_loss = float(np.mean(losses))
        avg_accuracy = float(np.mean(np.array(accuracies))) if accuracies else 0.0

        return avg_loss, {"FL_accuracy": avg_accuracy}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

class FedProxStrategy(Strategy):
    def __init__(
        self,
        num_clients: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        alpha_dirichlet: float,
        dataset_name: str,
        seed: int,
        mu: float,
    ):
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha_dirichlet = alpha_dirichlet
        self.dataset_name = dataset_name
        self.seed = seed
        self.mu = mu
        self.initial_parameters: Optional[Parameters] = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(self.num_clients)
        config: Dict[str, Scalar] = {"mu": float(self.mu)}
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        weights = []
        num_examples = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights.append(ndarrays)
            num_examples.append(fit_res.num_examples)

        if not weights:
            return None, {}

        total = sum(num_examples)
        avg_weights = [
            np.sum([weights[client_idx][i] * num_examples[client_idx] for client_idx in range(len(weights))], axis=0) / total
            for i in range(len(weights[0]))
        ]

        avg_loss = float(np.mean(np.array([fit_res.metrics["FL_loss"] for _, fit_res in results if "FL_loss" in fit_res.metrics])))
        accuracies = [fit_res.metrics["FL_accuracy"] for _, fit_res in results if "FL_accuracy" in fit_res.metrics]
        avg_accuracy = float(np.mean(np.array(accuracies))) if accuracies else 0.0

        return ndarrays_to_parameters(avg_weights), {"FL_loss": avg_loss, "FL_accuracy": avg_accuracy}

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.sample(self.num_clients)
        config = {}
        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        losses = [res.loss for _, res in results]
        accuracies = [res.metrics["FL_accuracy"] for _, res in results if "FL_accuracy" in res.metrics]

        avg_loss = float(np.mean(losses))
        avg_accuracy = float(np.mean(np.array(accuracies))) if accuracies else 0.0

        return avg_loss, {"FL_accuracy": avg_accuracy}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

class ScaffoldStrategy(Strategy):
    def __init__(
        self,
        num_clients: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        alpha_dirichlet: float,
        dataset_name: str,
        seed: int,
    ):
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha_dirichlet = alpha_dirichlet
        self.dataset_name = dataset_name
        self.seed = seed
        self.initial_parameters: Optional[Parameters] = None
        self.c: Optional[List[np.ndarray]] = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        if self.c is None and self.initial_parameters is not None:
            param_arrays = parameters_to_ndarrays(self.initial_parameters)
            self.c = [np.zeros_like(p) for p in param_arrays]
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Ensure self.c is initialized (even in the first round)
        if self.c is None:
            param_arrays = parameters_to_ndarrays(parameters)
            self.c = [np.zeros_like(p) for p in param_arrays]
        c_serialized = json.dumps([p.tolist() for p in self.c])
        config = {"c": c_serialized, "learning_rate": float(self.learning_rate)}
        clients = client_manager.sample(self.num_clients)
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        weights = []
        num_examples = []
        losses = []
        accuracies = []
        cks = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights.append(ndarrays)
            num_examples.append(fit_res.num_examples)
            # Collect metrics if present
            if "FL_loss" in fit_res.metrics:
                losses.append(fit_res.metrics["FL_loss"])
            if "FL_accuracy" in fit_res.metrics:
                accuracies.append(fit_res.metrics["FL_accuracy"])
            if "ck" in fit_res.metrics:
                ck_raw = fit_res.metrics["ck"]
                if isinstance(ck_raw, str):
                    ck = [np.array(arr) for arr in json.loads(ck_raw)]
                    cks.append(ck)

        # Aggregate weights
        total = sum(num_examples)
        avg_weights = [
            np.sum([weights[client_idx][i] * num_examples[client_idx] for client_idx in range(len(weights))], axis=0) / total
            for i in range(len(weights[0]))
        ]

        # Update c
        if cks and self.c is not None:
            avg_delta = [np.mean([ck[i] - self.c[i] for ck in cks], axis=0) for i in range(len(self.c))]
            self.c = [self.c[i] + avg_delta[i] for i in range(len(self.c))]

        # Compute average loss and accuracy
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")

        return ndarrays_to_parameters(avg_weights), {"FL_loss": avg_loss, "FL_accuracy": avg_accuracy}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.sample(self.num_clients)
        config = {}
        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        losses = [res.loss for _, res in results]
        accuracies = [res.metrics["FL_accuracy"] for _, res in results if "FL_accuracy" in res.metrics]

        avg_loss = float(np.mean(losses))
        avg_accuracy = float(np.mean(np.array(accuracies))) if accuracies else 0.0

        return avg_loss, {"FL_accuracy": avg_accuracy}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
