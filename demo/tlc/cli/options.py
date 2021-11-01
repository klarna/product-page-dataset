"""
Module containing cli options for the main entry points of the project.
"""

import pathlib
from pathlib import Path

import click

train_options = [
    click.option("--dataset", "-d", default="dataset", type=str, help="Path to training dataset folder."),
    click.option(
        "--optimizer-folder",
        "-o",
        default=".opt",
        type=Path,
        help="Path to folder containing an optimizer.pt file.",
    ),
    click.option(
        "--gin-config-path",
        "-g",
        multiple=True,
        default=[],
        help="Path to local gin config file. The parameters in the local gin config will override config.gin.",
    ),
    click.option("--no-cache", "-n", is_flag=True, help="Clear pickled dataset cache before running."),
]


evaluate_options = [
    click.option("--model-loc", "-m", type=pathlib.Path, required=True, help="Location of the model to evaluate."),
    click.option("--dataset", "-d", default="dataset", type=str, help="Path to dataset folder."),
    click.option("--dataset-cache", default="dataset_cache", type=str, help="Path to dataset_cache folder."),
    click.option("--prediction", "-p", is_flag=True, help="Return prediction accuracy."),
    click.option("--on-train-set", "-t", is_flag=True, help="Use training set instead of test set for evaluation"),
    click.option("--no-cache", "-c", is_flag=True, help="Clear pickled dataset cache before running."),
    click.option("--n-data", "-n", type=int, default=None, required=False, help="Number of data to use."),
    click.option(
        "--use-augmented", "-a", is_flag=True, help="Use predictions available in augmented daatset to evaluate."
    ),
]
