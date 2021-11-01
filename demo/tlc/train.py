"""
Entry point for tree-lstm classifier
"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import click
import gin.torch.external_configurables
import torch
from torch.utils.data import DataLoader

from .cli.decorators import add_options
from .cli.options import train_options
from .dataset.loaders import PicklingTreeDataset, sample_nodes_from_trees
from .device import get_torch_device
from .evaluate import eval_on_test_set, predict_on_test_set
from .models import elementclassifiers as classifiers
from .structures import trees
from .trainers import LoggingTrainer
from .utilities import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


@gin.configurable
@dataclass
class ExperimentParameters:
    """
    Data class to encapsulate experiment parameters.
    """

    run_name: str
    model_id: str
    tree_type: str
    n_train_data: int
    n_test_data: int
    n_features: int
    num_workers: int


@gin.configurable
@dataclass
class HyperParameters:
    """
    Data class to encapsulate common hyper parameters.
    """

    batch_size: int
    n_positive_samples: int
    n_negative_samples: int
    latent_dimension: int
    n_epochs: int
    train_test_split_ratio: float
    optimizer: str
    optimizer_parameters: Dict
    loss_function: Any


@gin.configurable(allowlist=["hyper_parameters", "experiment_parameters"])
def run(
    dataset: str,
    optimizer_folder: str,
    experiment_parameters: ExperimentParameters,
    hyper_parameters: HyperParameters,
    no_cache=False,
):
    """
    Start a training run.

    :param dataset: Path to dataset.
    :param optimizer_folder: Path to folder containing an optimizer.
    :param experiment_parameters: ExperimentParameters object containing parameters to run experiment with.
    :param hyper_parameters:  HyperParameters object containing parameters to run experiment with.
    :param no_cache: If set to true, the data tree cache will be cleared before starting training.
    :return:
    """

    run_location = Path("runs") / experiment_parameters.run_name
    run_location.mkdir(parents=True, exist_ok=True)

    device = get_torch_device()  # pylint: disable=no-value-for-parameter

    model = None
    model_file = Path(experiment_parameters.model_id)
    if model_file.is_file():
        model = torch.load(model_file)
    else:
        try:
            classifier = getattr(classifiers, experiment_parameters.model_id)
        except AttributeError as e:
            raise AttributeError(f"No model called {experiment_parameters.model_id} exists!") from e
    try:
        tree_type_class = getattr(trees, experiment_parameters.tree_type)
    except AttributeError as e:
        raise AttributeError(f"No tree called {experiment_parameters.tree_type} exists!") from e

    feature_length = experiment_parameters.n_features
    n_labels = len(classifier.ALL_LABELS)
    if not model:
        model = classifier(feature_length, n_labels, hyper_parameters.latent_dimension)
    model.to(device=device)
    model.initialize()

    try:
        collate_fn = model.collate_fn()
    except AttributeError:
        collate_fn = sample_nodes_from_trees(hyper_parameters.n_positive_samples, hyper_parameters.n_negative_samples)

    train_set, validation_set = PicklingTreeDataset(
        location=dataset + "/train",
        cache_location="dataset_cache/train",
        n_data=experiment_parameters.n_train_data,
        tree_type=tree_type_class,
        clean=no_cache,
    ).split(hyper_parameters.train_test_split_ratio)

    train_loader = DataLoader(
        train_set,
        batch_size=hyper_parameters.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=experiment_parameters.num_workers,
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=hyper_parameters.batch_size,
        collate_fn=collate_fn,
        num_workers=experiment_parameters.num_workers,
    )

    test_set = PicklingTreeDataset(
        location=dataset + "/test",
        cache_location="dataset_cache/test",
        n_data=experiment_parameters.n_test_data,
        tree_type=tree_type_class,
        clean=no_cache,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=hyper_parameters.batch_size,
        collate_fn=collate_fn,
        num_workers=experiment_parameters.num_workers,
    )

    optimizer_file = Path(optimizer_folder) / "optimizer.pt"
    if optimizer_file.is_file():
        optimizer = torch.load(optimizer_file)
    else:
        try:
            opt = getattr(torch.optim, hyper_parameters.optimizer)
            optimizer = opt(model.parameters(), **hyper_parameters.optimizer_parameters)
        except AttributeError as e:
            raise AttributeError(f"No optimizer called {hyper_parameters.optimizer} exists!") from e

    loss_fn = hyper_parameters.loss_function

    trainer = LoggingTrainer(model, optimizer, loss_fn, run_location)

    with open(run_location / "simulation_data.txt", "w") as fhandle:
        fhandle.write(
            f"""
Full command: {" ".join(sys.argv)}
Dataset location: {train_set.location}
    N samples (train): {len(train_set.snapshots)}
    N samples (validation): {validation_loader and len(validation_set.snapshots)}
    N samples (test): {len(test_set.snapshots)}
    N features: {feature_length}
    N labels: {n_labels}
Model: {model}
Optimizer: {optimizer}
Trainer: {str(trainer)}
Loss: {loss_fn}"""
        )

    trainer.train(train_loader, hyper_parameters.n_epochs, validation_loader)

    with open(run_location / "timing_data.txt", "w") as fhandle:
        fhandle.write(
            f"""Training wallclock time:
    Forward: {trainer.timing["forward"]}
    Backward: {trainer.timing["backward"]}
    Validation: {trainer.timing["validation"]}"""
        )

    with open(run_location / "classification_report.txt", "w") as fhandle:
        fhandle.write(eval_on_test_set(model, test_loader))

    prediction_report, distance_report = predict_on_test_set(model, test_set, [1, 3, 5])

    with open(run_location / "prediction_report.txt", "w") as fhandle:
        fhandle.write(str(prediction_report))

    with open(run_location / "distance_report.txt", "w") as fhandle:
        fhandle.write(str(distance_report))


@add_options(train_options)
@click.command()
def main(
    dataset: str,
    optimizer_folder: str,
    gin_config_path: List[str],
    no_cache: bool,
):

    gin.parse_config_file("gin_configs/config.gin")
    for _conf in gin_config_path:
        logger.info(f"Loading {_conf} config file; any existing parameter will be overwritten!")
        gin.parse_config_file(_conf)

    run(dataset, optimizer_folder, no_cache=no_cache)  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
