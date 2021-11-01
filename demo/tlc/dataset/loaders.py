"""Interfaces to snaphots datasets"""

from __future__ import annotations

import copy
import glob
import json
import os
import pickle
import shutil
from pathlib import Path
from random import shuffle
from typing import Callable, Collection, Iterator, List, Optional, Tuple, Type, Union

import bs4
import webtraversallibrary as wtl
from torch.utils.data import Dataset

from ..structures.trees import DataTree
from ..utilities import stream_sample


class TreeDatasetException(Exception):
    pass


class TreeDataset(Dataset):
    """
    Map style dataset returning DataTrees. Constructor should point to a folder
    containing folders of wtl snapshots.
    """

    def __init__(
        self,
        tree_type: Type[DataTree] = DataTree,
        location: Optional[Union[Path, str]] = None,
        n_data: Optional[int] = None,
    ):
        self.location = location
        self.snapshots = []
        self.tree_type = tree_type

        if self.location is not None:
            self.location = Path(location)
            self.snapshots = sorted(list(glob.glob(f"{str(self.location)}/*/*/*")), key=os.path.basename)[:n_data]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int) -> DataTree:
        snapshot_location = Path(self.snapshots[index])
        with open(snapshot_location / "elements_metadata.json") as f:
            element_metadata = json.load(f)
        with open(snapshot_location / "page_metadata.json") as f:
            page_metadata = json.load(f)

        # This seems to leak shared memory and file pointers when used in multiprocessing
        with open(snapshot_location / "source.html") as f:
            bs4object = bs4.BeautifulSoup(f, "html5lib")
        # bs4object = None

        snapshot = wtl.PageSnapshot(bs4object, page_metadata, element_metadata)

        return self.tree_type.from_snapshot(snapshot)

    def __iter__(self) -> Iterator[DataTree]:
        for i in range(len(self)):
            yield self[i]

    def split(self, ratio: float, deepcopy=False) -> Tuple[TreeDataset, TreeDataset]:
        """Split the dataset according to the given ratio.

        The dataset will be shallow copied unless `deepcopy` is True."""

        # Create the splits by copying the dataset
        if not deepcopy:
            a = copy.copy(self)
            b = copy.copy(self)
        else:
            a = copy.deepcopy(self)
            b = copy.deepcopy(self)

        # Number of samples in a
        n_a = int(ratio * len(self))

        # Create new lists of snapshot for the splits
        a.snapshots = a.snapshots[:n_a]
        b.snapshots = b.snapshots[n_a:]

        return a, b


class PicklingTreeDataset(TreeDataset):
    """
    Map style dataset returning DataTrees. Constructor should point to a folder
    containing folders of wtl snapshots. When first loaded, the DataTrees are pickled
    in the folder specified as `cache_location`.
    """

    def __init__(
        self,
        tree_type: Type[DataTree] = DataTree,
        location: Optional[Union[Path, str]] = None,
        cache_location: Union[Path, str] = "dataset_cache/train",
        n_data: Optional[int] = None,
        clean: bool = False,
    ):
        super().__init__(location=location, tree_type=tree_type, n_data=n_data)
        self.cache_location = Path(cache_location)
        if clean:
            answer = ""
            while answer not in ["y", "n"]:
                answer = input("Are you sure you want to delete dataset cache?").lower()
            if answer == "y":
                shutil.rmtree(self.cache_location)
        self.cache_location.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, idx: int) -> DataTree:
        snapshot_name = self.snapshots[idx]
        try:
            with open(self.cache_location / (snapshot_name + ".pkl"), "rb") as fhandle:
                tree = pickle.load(fhandle)

                # We use a direct type check because we want do detect different subclasses
                if type(tree) != self.tree_type:  # pylint: disable=unidiomatic-typecheck
                    raise FileNotFoundError

            return tree
        except FileNotFoundError:
            treefile = super().__getitem__(idx)
            os.makedirs(self.cache_location / os.path.dirname(snapshot_name), exist_ok=True)
            with open(self.cache_location / (snapshot_name + ".pkl"), "wb") as fhandle:
                pickle.dump(treefile, fhandle)
            return treefile


def sample_nodes_from_trees(
    n_positive_samples: int, n_negative_samples: int
) -> Callable[[Collection[DataTree]], List[DataTree]]:
    """Returns a collate_fn that assembles batches of nodes from batches of DataTrees"""

    def _collate_fn(batch: Collection[DataTree]) -> List[DataTree]:
        training_nodes = []
        for tree in batch:
            if tree is None:
                continue
            training_nodes.extend(stream_sample(tree.labeled, n_positive_samples))
            training_nodes.extend(stream_sample(tree.unlabeled, n_negative_samples))

        shuffle(training_nodes)
        return training_nodes

    return _collate_fn
