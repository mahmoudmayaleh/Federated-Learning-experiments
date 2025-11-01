import flwr as fl
import torch
import numpy as np
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    FitIns, FitRes, Status, Code,
    EvaluateIns, EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import List
from model import CustomFashionModel

class CustomClient(fl.client.Client):
    def __init__(self, model: torch.nn.Module, train_loader, test_loader, device: torch.device) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.global_weights = None  # Initialize global_weights

    def get_properties(self, instruction: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={"device": str(self.device)}
        )

    def get_parameters(self, instruction: GetParametersIns) -> GetParametersRes:
        params = self.get_model_parameters()
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params)
        )

    def fit(self, instruction: FitIns) -> FitRes:
        params = parameters_to_ndarrays(instruction.parameters)
        self.set_model_parameters(params)
        lr = float(str(instruction.config.get("learning_rate", 0.01)))
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1):  # set to your desired local epochs
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                # Proximal term for FedAvg
                proximal_term = 0.0
                mu = float(instruction.config.get("mu", 0.0)) 
                if mu > 0 and self.global_weights is not None:
                    for w, w0 in zip(self.model.parameters(), self.global_weights):
                        proximal_term += ((w - w0.to(self.device)) ** 2).sum()
                
                output = self.model(data)
                loss = criterion(output, target) + (mu / 2) * proximal_term
                loss.backward()
                optimizer.step()
        updated_weights = self.get_model_parameters()

        # Calculate loss and accuracy for metrics
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        avg_loss = loss / total
        accuracy = correct / total

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(updated_weights),
            num_examples=len(self.train_loader.dataset),
            metrics={
                "FL_loss": float(avg_loss),
                "FL_accuracy": float(accuracy),
            },
        )

    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        params = parameters_to_ndarrays(instruction.parameters)
        self.set_model_parameters(params)

        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        avg_loss = loss / total
        accuracy = correct / total
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(avg_loss),
            num_examples=len(self.test_loader.dataset),
            metrics={"FL_accuracy": float(accuracy)},
        )

    def to_client(self) -> "CustomClient":
        return self

    def get_model_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict)