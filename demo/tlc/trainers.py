"""Training objects"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from typing import Collection, Optional, Tuple, Union

import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .device import get_torch_device
from .models.prototypes import TreeClassifier
from .structures.trees import DataTree, NodePair


class TimeField:
    """Class to store training time information"""

    def __init__(self, start_value: float = 0.0, start_count: int = 0):
        self.data = start_value
        self.ndata = start_count

    def __add__(self, info: Tuple[float, int]) -> TimeField:
        self.data += info[0]
        self.ndata += info[1]
        return self

    def __repr__(self) -> str:
        if self.ndata == 0:
            return "self.ndata is 0"
        return f"{self.data} / {self.ndata} = {self.data/self.ndata}"


class SimpleTrainer:
    """Main class for model training"""

    def __init__(self, model: TreeClassifier, optimizer: torch.optim.Optimizer, loss_fn: _Loss):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_epoch = 0
        self.timing = {  # Store timing information for training steps
            "forward": TimeField(),
            "backward": TimeField(),
            "validation": TimeField(),
        }
        self.device = get_torch_device()  # pylint: disable=no-value-for-parameter

    def _model_forward(self, batch: Collection[Union[DataTree, NodePair]]) -> torch.Tensor:
        """Return the loss of the model run on a batch of nodes"""
        t0 = time()
        y_logit = list(self.model.forward_batch(batch))
        y_target = [self.model.label_encode(node.label) for node in batch]
        self.timing["forward"] += (time() - t0, len(batch))
        self.model.reset()

        if not y_logit and not y_target:
            return torch.tensor(0, device=self.device)  # pylint: disable=not-callable

        return self.loss_fn(
            torch.stack(y_logit), torch.tensor(y_target, device=self.device)  # pylint: disable=not-callable
        )

    def _training_step(self, batch: Collection[Union[DataTree, NodePair]]) -> Tuple[float, float]:
        """Run one step of training on a batch of sampled nodes"""

        loss = self._model_forward(batch)

        # Compute the gradients for this prediction batch
        t0 = time()
        loss.backward()
        self.timing["backward"] += (time() - t0, 1)

        # Compute the sum of the normalized gradients to track exploding/disappearing gradients.
        grad_norm = 0.0
        for param in self.model.parameters():
            grad_norm += param.data.norm(2).item() ** 2

        grad_norm = grad_norm ** 0.5

        # Commit the gradients to the weights
        self.optimizer.step()

        # Reset gradient buffer
        self.optimizer.zero_grad()

        return loss.item(), grad_norm

    def train(
        self,
        train_loader: DataLoader,
        n_epochs: int,
        validation_loader: Optional[DataLoader] = None,
    ) -> None:
        """Train the model on the batches provided by the train_loader for a given number of epochs; optionally, use a
        validation_loader for tracking validation_error"""

        validation_loss = float("inf")

        with TemporaryDirectory() as _tmp_dir:

            for epoch in tqdm(range(n_epochs), position=0, leave=True, dynamic_ncols=True, desc="Training epoch"):

                self.model.train()

                for batch in tqdm(train_loader, position=1, leave=False, dynamic_ncols=True, desc="Training batch"):
                    self._training_step(batch)

                if validation_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        current_validation_loss = 0.0
                        for batch in validation_loader:
                            current_validation_loss += self._model_forward(batch).item()
                            if current_validation_loss < validation_loss:
                                validation_loss = current_validation_loss
                                self.best_epoch = epoch
                                self.model.save(loc=_tmp_dir)

            torch.save(self.optimizer, _tmp_dir + "/optimizer.pt")

            if validation_loader is not None:
                self.model.load(loc=_tmp_dir)


class LoggingTrainer(SimpleTrainer):
    """Class for model training with tensorboard interface"""

    def __init__(self, model: TreeClassifier, optimizer: torch.optim.Optimizer, loss_fn: _Loss, run_location: Path):
        super().__init__(model, optimizer, loss_fn)
        # self.logboard = get_board()
        self.model_location = run_location

    def train(
        self,
        train_loader: DataLoader,
        n_epochs: int,
        validation_loader: Optional[DataLoader] = None,
    ) -> None:

        validation_loss = float("inf")

        for epoch in tqdm(range(n_epochs), position=0, leave=True, dynamic_ncols=True, desc="Training epoch"):

            training_loss = 0.0
            training_gradnorm = 0.0
            self.model.train()
            for batch in tqdm(train_loader, position=1, leave=False, dynamic_ncols=True, desc="Training batch"):
                loss, gradnorm = self._training_step(batch)

                training_loss += loss
                training_gradnorm += gradnorm

            # self.logboard.training_graph.add(training_loss / len(train_loader.dataset))  # type:ignore
            # self.logboard.gradient_norm_graph.add(training_gradnorm / len(train_loader.dataset))  # type:ignore

            self.model.eval()
            if validation_loader is not None:
                with torch.no_grad():
                    current_validation_loss = 0.0
                    t0 = time()
                    for batch in validation_loader:
                        current_validation_loss += self._model_forward(batch).item()
                    self.timing["validation"] += (time() - t0, len(validation_loader.dataset))  # type:ignore

                    current_validation_loss /= len(validation_loader.dataset)  # type:ignore

                    # self.logboard.validation_graph.add(current_validation_loss)
                    if current_validation_loss < validation_loss:
                        self.best_epoch = epoch
                        validation_loss = current_validation_loss

                        self.model.to(torch.device("cpu"))
                        self.model.save(loc=self.model_location)
                        self.model.to(get_torch_device())

        torch.save(self.optimizer, self.model_location / "optimizer.pt")

        if validation_loader is not None:
            self.model.load(loc=self.model_location)
