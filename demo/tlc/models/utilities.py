"""Various model utilities"""

from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, Union

import torch

TOKEN_CHARACTER_PADDING = "<CPAD>"
TOKEN_WORD_PADDING = "<WPAD>"


embeddings_dict = None

_T = TypeVar("_T")


class SaveMixin:
    """A mixin that enables saving and loading of models"""

    def save(self: _T, loc: Union[str, Path] = ".models", name: str = "model.pt") -> None:
        location = Path(loc)
        location.mkdir(parents=True, exist_ok=True)
        torch.save(self, location / name)

    @classmethod
    def load(cls: Type[_T], loc: Union[str, Path] = ".model", name: str = "model.pt") -> _T:
        location = Path(loc)
        return torch.load(location / name)


def list_indexer(my_list: List[Any]) -> Dict[Any, int]:
    """
    Function that creates a dictionary index from a list.

    :param my_list: List to create the index dictionary for.
    :return: Dictionary with list items as keys and index in list as values.
    """
    return dict(zip(my_list, range(len(my_list))))
