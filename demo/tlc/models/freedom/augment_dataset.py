"""
Entrypoint for augmenting dataset for stage 2 of FreeDOM.
"""
import json
import logging
import pathlib
import pickle
from itertools import zip_longest
from pathlib import Path

import click
import numpy
import torch
from tqdm import tqdm

from tlc.cli.decorators import add_options
from tlc.dataset.loaders import PicklingTreeDataset
from tlc.device import get_torch_device, reset_device
from tlc.models.prototypes import TreeClassifier
from tlc.structures import trees

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

options = [
    click.option("--dataset", "-d", default="dataset", type=str, help="Path to training dataset folder."),
    click.option("--model_path", "-m", type=str, help="Path to trained model."),
    click.option("--tree-type", "-t", type=str, help="Tree type to use with the model."),
    click.option("--n-train-data", "-n", type=int, help="Number of data tree's to augment in training set."),
    click.option("--n-test-data", "-n", type=int, help="Number of data tree's to augment in test set."),
    click.option("--tree-batch-size", "-b", type=int, help="Tree batch size."),
    click.option("--gpu", "-g", is_flag=True, type=bool, help="Use gpu or not."),
]


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def augment_dataset(tree_dataset: PicklingTreeDataset, model: TreeClassifier, tree_batch_size: int):
    """
    Enriches dataset with a tentative label, probabilities and node embedding.

    :param tree_dataset: Tree datasset to augment.
    :param model: Model to use for augmenting dataset.
    :param tree_batch_size: Size of tree batches to load together.
    """
    failed_batches = []
    for batch_index, batches in tqdm(
        enumerate(grouper(enumerate(tree_dataset), tree_batch_size)),
        position=0,
        leave=True,
        dynamic_ncols=True,
        desc="Augmenting dataset",
        total=int(len(tree_dataset) / tree_batch_size),
    ):

        batches = [tupe for tupe in batches if tupe[1]]

        flat_list = [item for sublist in batches for item in sublist[1]]

        try:
            node_embedding_batch = model.tree_model.node_embedding_batch(list(flat_list))  # type: ignore
            tree_model_output = model.tree_model.dropout_layer(  # type: ignore
                model.tree_model.local_node_mlp(node_embedding_batch)  # type: ignore
            )
            class_probabilities_batch = model.softmax(model.output(tree_model_output)).tolist()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM at batch:", batch_index)
                failed_batches.append(
                    {
                        "batch_index": batch_index,
                        "trees_in_batch": [tree_dataset.snapshots[tree_index] for tree_index, _ in batches],
                    }
                )
                continue
            raise e

        tentative_labels = [
            model.label_decode(int(numpy.argmax(class_probabilities)))
            for class_probabilities in class_probabilities_batch
        ]

        for tree_index, tree in batches:

            tree_size = tree.size

            class_probabilities_batch_tree = class_probabilities_batch[:tree_size]
            node_embedding_batch_tree = node_embedding_batch[:tree_size]
            tentative_labels_tree = tentative_labels[:tree_size]

            class_probabilities_batch = class_probabilities_batch[tree_size:]
            node_embedding_batch = node_embedding_batch[tree_size:]
            tentative_labels = tentative_labels[tree_size:]

            snapshot_location = pathlib.Path(tree_dataset.snapshots[tree_index])

            for node, class_probabilities, node_embedding, tentative_label in tqdm(
                zip(tree, class_probabilities_batch_tree, node_embedding_batch_tree, tentative_labels_tree),
                position=1,
                leave=False,
                dynamic_ncols=True,
                desc="Augmenting Tree",
                total=tree.size,
            ):

                freedom_metadata = {
                    "tentative_label": tentative_label,
                    "node_embedding": node_embedding.tolist(),
                    "class_probabilities": class_probabilities,
                }

                node.feature_vector.feature_tentative_label = freedom_metadata["tentative_label"]
                node.feature_vector.feature_class_probabilities = freedom_metadata["class_probabilities"]
                node.feature_vector.feature_node_embedding = freedom_metadata["node_embedding"]

            # Update tree cache with new data.
            with open(tree_dataset.cache_location / (str(snapshot_location) + ".pkl"), "wb") as fhandle:
                pickle.dump(tree, fhandle)

    # Return failed batches to re-run.
    return failed_batches


@add_options(options)
@click.command()
def run_augmentation(
    dataset: str,
    model_path: pathlib.Path,
    tree_type: str,
    n_train_data: int,
    n_test_data: int,
    tree_batch_size: int,
    gpu: bool,
):
    """
    Main entrypoint
    """
    logger.info(f"Starting to augment dataset at {str(dataset)}")
    device = get_torch_device(gpu)
    logger.info(f"Using device: f{device}")

    model_file = pathlib.Path(model_path)
    if model_file.is_file():
        pretrained_tree_classifier: TreeClassifier = torch.load(model_file)
    else:
        raise AttributeError("TreePairClassifier needs a pre-trained tree classifier to be initialized.")

    try:
        tree_type_class = getattr(trees, tree_type)
    except AttributeError as e:
        raise AttributeError(f"No tree called {tree_type} exists!") from e

    pretrained_tree_classifier.to(device=device)
    pretrained_tree_classifier.eval()

    train_set = PicklingTreeDataset(
        location=dataset + "/train",
        cache_location="dataset_cache/train",
        n_data=n_train_data,
        tree_type=tree_type_class,
    )

    test_set = PicklingTreeDataset(
        location=dataset + "/test",
        cache_location="dataset_cache/test",
        n_data=n_test_data,
        tree_type=tree_type_class,
    )

    with torch.no_grad():
        logger.info("Augmenting train dataset")
        train_failed_batches = augment_dataset(train_set, pretrained_tree_classifier, tree_batch_size)

        Path("run_logs/").mkdir(parents=True, exist_ok=True)
        with open("run_logs/train_set_augmentation_failed_batches.json", "w") as jsonf:
            json.dump(train_failed_batches, jsonf)

        logger.info("Augmenting test dataset")
        test_failed_batches = augment_dataset(test_set, pretrained_tree_classifier, tree_batch_size)

        Path("run_logs/").mkdir(parents=True, exist_ok=True)
        with open("run_logs/test_set_augmentation_failed_batches.json", "w") as jsonf:
            json.dump(test_failed_batches, jsonf)

        if train_failed_batches:
            logger.info("Augmenting failed batches from train dataset using memory")
            reset_device()
            device = get_torch_device(False)
            logger.info(f"Using device: f{device}")
            pretrained_tree_classifier.to(device=device)
            failed_train_set = PicklingTreeDataset(
                location=dataset + "/train",
                cache_location="dataset_cache/train",
                n_data=n_train_data,
                tree_type=tree_type_class,
            )
            failed_train_set.snapshots = [item for batch in train_failed_batches for item in batch["trees_in_batch"]]
            augment_dataset(failed_train_set, pretrained_tree_classifier, tree_batch_size)

        if test_failed_batches:
            logger.info("Augmenting failed batches from test dataset using memory.")
            reset_device()
            device = get_torch_device(False)
            logger.info(f"Using device: f{device}")
            pretrained_tree_classifier.to(device=device)
            failed_test_set = PicklingTreeDataset(
                location=dataset + "/test",
                cache_location="dataset_cache/test",
                n_data=n_test_data,
                tree_type=tree_type_class,
            )
            failed_test_set.snapshots = [item for batch in test_failed_batches for item in batch["trees_in_batch"]]
            augment_dataset(failed_test_set, pretrained_tree_classifier, tree_batch_size)


if __name__ == "__main__":
    run_augmentation()  # pylint: disable=no-value-for-parameter
