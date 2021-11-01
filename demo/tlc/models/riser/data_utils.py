"""Utilities for data handling"""

import pathlib
import pickle
import sys
from typing import Any, List, Tuple

import numpy as np
import torch
from webtraversallibrary import PageElement

from tlc.models import riser
from tlc.structures.mixins import FeatureMixin

from .data import Document, Vocabulary

sys.modules["riser"] = riser
dir_path = pathlib.Path(__file__).resolve().parent / "vocabs/"
PATH_COMPONENT_VOCAB_PATH = pathlib.Path(dir_path / "path_component_vocab.p")
WORD_VOCAB_PATH = pathlib.Path(dir_path / "word_vocab.p")

try:
    with open(WORD_VOCAB_PATH, "rb") as f:
        word_vocab: Vocabulary = pickle.load(f)
    with open(PATH_COMPONENT_VOCAB_PATH, "rb") as f:
        path_component_vocab: Vocabulary = pickle.load(f)
except FileNotFoundError as e:
    print("Vocabulary files not found: %s", str(e))


class RiSERFeatureMixin(FeatureMixin):
    """Features used by the RiSER model"""

    @staticmethod
    def _features(element: PageElement) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        tag = element.tag
        input_features = Document(tag.html, 50, 20)
        if not input_features.words:
            input_features.words = ["0"]
        if not input_features.paths:
            input_features.paths = [["0"]]
        return (
            list_to_tensor([word_vocab[w] for w in input_features.words]).long(),
            list_to_tensor([[path_component_vocab[c] for c in p] for p in input_features.paths]).long(),
        )


def list_to_tensor(my_list: List[Any]) -> torch.Tensor:
    """Wrapper for converting a nested list of numbers or tensors into a tensors of floats."""
    return torch.from_numpy(np.array(my_list))
