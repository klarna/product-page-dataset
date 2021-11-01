"""
Entry point for tree-lstm element classifier
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Tuple, Union

import click
import gin.torch.external_configurables
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from tlc.cli.decorators import add_options
from tlc.cli.options import evaluate_options
from tlc.device import get_torch_device
from tlc.structures import trees

from .dataset.loaders import PicklingTreeDataset, TreeDataset, sample_nodes_from_trees
from .utilities import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

if TYPE_CHECKING:
    from .models.prototypes import TreeClassifier, TreePredictType
    from .structures.trees import DataTree


def eval_on_test_set(model: TreeClassifier, test_loader: DataLoader) -> str:
    """Return a classification report for the model on the test set"""

    y_pred: List[str] = []
    y_true: List[str] = []

    logger.info("Computing classification report.")
    for batch in tqdm(test_loader):

        with torch.no_grad():
            y_pred.extend(str(model.predict(node, human_readable=True)) for node in batch)

        y_true.extend(node.label for node in batch)

        model.reset()

    return classification_report(y_true, y_pred, digits=4) if test_loader else ""


def evaluate_prediction(
    predictions: TreePredictType, n_candidates: List[int]
) -> Tuple[Dict[Union[str, int], Counter], Dict[Union[str, int], Counter]]:

    """Return statistics of the predictions returned by a predict_tree method"""

    n_correct: DefaultDict[Union[str, int], Counter] = defaultdict(Counter)
    n_total: DefaultDict[Union[str, int], Counter] = defaultdict(Counter)

    for prediction, candidates in predictions.items():
        for top_n in n_candidates:

            # Check if any of the top_n candidate matches the true label
            candidate_matches = any(prediction == node.label for _, node in candidates[:top_n])

            n_correct[prediction][top_n] += int(candidate_matches)
            n_total[prediction][top_n] += 1

    return n_correct, n_total


def predict_on_test_set(
    model: TreeClassifier, test_set: TreeDataset, n_candidates: List[int], use_augmented: bool = False
) -> Tuple[str, str]:
    """Return a prediction report for the Top N candidates on the test set"""

    n_candidates.sort()

    total_correct: Dict[Union[str, int], Counter] = defaultdict(Counter)
    total_total: Dict[Union[str, int], Counter] = defaultdict(Counter)

    distances_to_true_node: Dict[Union[str, int], List[int]] = defaultdict(list)

    failed_trees = []

    logger.info("Computing prediction report.")
    for tree_index, tree in tqdm(enumerate(test_set), total=len(test_set)):
        if tree is None:
            continue

        try:
            with torch.no_grad():
                predictions = model.predict_tree(tree, max(n_candidates), use_augmented=use_augmented)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM at Tree:", tree_index)
                failed_trees.append({"tree_index": tree_index, "tree_path": test_set.snapshots[tree_index]})
                continue
            raise e

        distances = compute_distance_to_true_nodes(predictions, tree)
        for label, distance in distances.items():
            distances_to_true_node[label].append(distance)

        n_correct, n_total = evaluate_prediction(predictions, n_candidates)

        update_counts(total_correct, n_correct)
        update_counts(total_total, n_total)

        model.reset()

    # Save failed trees to re-run.
    Path("run_logs/").mkdir(parents=True, exist_ok=True)
    with open("run_logs/prediction_failed_trees.json", "w") as jsonf:
        json.dump(failed_trees, jsonf)

    prediction_format = "{:>12}" + "{:^10}" * (len(n_candidates) + 1)
    prediction_report = [prediction_format.format("Class", *(f"Top {i}" for i in n_candidates), "Support")]

    correct_out: List[int] = []
    total_out: List[int] = []

    for label in total_correct:
        if label == "unlabeled":
            continue

        correct = [total_correct[label][i] for i in n_candidates]
        total = [total_total[label][i] for i in n_candidates]

        correct_out = [i + j for i, j in zip_longest(correct, correct_out, fillvalue=0)]
        total_out = [i + j for i, j in zip_longest(total, total_out, fillvalue=0)]

        prediction_report.append(
            prediction_format.format(label, *(f"{c / t:0.4f}" for c, t in zip(correct, total)), f"({total[0]})")
        )

    prediction_report.append(
        prediction_format.format(
            "Avg.", *(f"{c / t:0.4f}" for c, t in zip(correct_out, total_out)), f"({total_out[0]})"
        )
    )

    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    distance_format = "{:>12}" + "{:^10}" * (len(quantiles))

    distance_report = [distance_format.format("Quantile", *(str(q) for q in quantiles))]
    for label, data in distances_to_true_node.items():
        distance_report.append(distance_format.format(label, *(str(round(np.quantile(data, q))) for q in quantiles)))

    return "\n".join(prediction_report), "\n".join(distance_report)


def compute_distance_to_true_nodes(predictions: TreePredictType, tree: DataTree) -> Dict[Union[str, int], int]:

    """Compute distances of the predicted nodes to the true nodes in the tree"""

    distances: Dict[Union[str, int], int] = dict()

    for node in tree.labeled:
        predicted_node = predictions[node.label][0][1]  # Highest predicted node for this class
        distances[node.label] = tree.distance(node, predicted_node)

    return distances


def update_counts(
    counter1: Dict[Union[str, int], Counter], counter2: Dict[Union[str, int], Counter]
) -> Dict[Union[str, int], Counter]:
    """In-place udpate of count dictionaries; clobbers the contents of counter1."""

    for label, counts in counter2.items():
        counter1[label].update(counts)

    return counter1


@add_options(evaluate_options)
@click.command()
@gin.configurable(
    allowlist=["batch_size", "n_positive_samples", "n_negative_samples", "n_data", "num_workers", "tree_type"]
)
def main(
    model_loc,
    dataset,
    dataset_cache,
    on_train_set,
    prediction,
    batch_size,
    n_positive_samples,
    n_negative_samples,
    n_data,
    num_workers,
    tree_type,
    no_cache,
    use_augmented,
):
    device = get_torch_device()  # pylint: disable=no-value-for-parameter

    try:
        tree_type_class = getattr(trees, tree_type)
    except AttributeError as e:
        raise AttributeError(f"No tree called {tree_type} exists!") from e

    location = Path(dataset) / ("train" if on_train_set else "test")
    cache_location = Path(dataset_cache) / ("train" if on_train_set else "test")
    data_set = PicklingTreeDataset(
        location=location, cache_location=cache_location, tree_type=tree_type_class, n_data=n_data, clean=no_cache
    )

    test_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=sample_nodes_from_trees(n_positive_samples, n_negative_samples),
        num_workers=num_workers,
    )

    model_file = model_loc / "model.pt"
    if model_file.is_file():
        model = torch.load(model_file)
        model.to(device)
    else:
        raise AttributeError(f"No model found in {model_loc}!")
    if prediction:
        prediction_report, distance_report = predict_on_test_set(
            model, data_set, n_candidates=[1, 3, 5], use_augmented=use_augmented
        )

        print("Prediction Report:")
        print(prediction_report)
        print("\nDistance Report:")
        print(distance_report)

    else:
        print(eval_on_test_set(model, test_loader))


if __name__ == "__main__":
    if Path("gin_configs/local_evaluate_config.gin").is_file():
        gin.parse_config_file("gin_configs/local_evaluate_config.gin")
    else:
        gin.parse_config_file("gin_configs/evaluate_config.gin")
    main()  # pylint: disable=no-value-for-parameter
