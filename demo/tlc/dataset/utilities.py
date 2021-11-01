"""Various utilities for dataset handling
TODO: A lot of this is deprecated and need to be cleaned up or removed"""
import ast
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, SupportsFloat, Tuple

import numpy as np
import torch
from tqdm import tqdm

METADATA_OF_INTEREST = (
    "tag",
    "font_weight",
    "font_size",
    "num_imgs",
    "num_svgs",
    "visibility",
    "size",
    "location",
)

DATAPOINT_LABEL_COLUMN = "xxxxx-ai-label"
KLARNAI_ID_COLUMN = "wtl-uid"
KLARNAI_PARENT_COLUMN = "wtl-parent-uid"

TAGS_OF_INTEREST = (
    "UNK",
    "a",
    "article",
    "aside",
    "b",
    "body",
    "br",
    "button",
    "canvas",
    "dd",
    "del",
    "div",
    "dl",
    "dt",
    "em",
    "fieldset",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "i",
    "iframe",
    "img",
    "input",
    "label",
    "li",
    "link",
    "main",
    "nav",
    "ol",
    "optgroup",
    "option",
    "p",
    "script",
    "section",
    "select",
    "span",
    "strong",
    "style",
    "table",
    "tbody",
    "td",
    "template",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "u",
    "ul",
)

LABELS_OF_INTEREST = [
    "name",
    "cart",
    "price",
    "addtocart",
    "mainpicture",
    #    "subjectnode", # No subjectnode in current dataset
]

TAG_TO_INT = dict((t, i) for i, t in enumerate(TAGS_OF_INTEREST))
COUNTRY_TO_LANGUAGE = {"DE": "de", "AT": "de", "SE": "se", "US": "en", "GB": "en", "FI": "fi", "NO": "no", "NL": "nl"}


def pre_process(text: str) -> str:
    """Basic cleaning of texts."""

    # make text lowercase.
    text = text.lower()

    # remove html markup.
    text = re.sub("(<.*?>)", "", text)

    # remove non-ascii and digits
    text = re.sub("(\\W|\\d)", " ", text)

    # remove whitespace.
    text = text.strip()

    # remove extra whitespaces
    text = " ".join(text.split())

    return text


VISIBILITY_TO_INT = dict((v, i) for i, v in enumerate(("hidden", "visible", "collapse", "inherit", "initial", "unset")))

IS_ACTIVE_TO_FLOAT = {
    True: 0.0,
    False: 1.0,
}


def bounding_box(loc: Dict[str, SupportsFloat], size: Dict[str, SupportsFloat]) -> np.ndarray:
    """Compute the bounding box from location and size"""
    x = float(loc["x"])
    y = float(loc["y"])
    width = float(size["width"])
    height = float(size["height"])
    return np.asarray([x, y, width, height]).reshape((4, 1))


def one_hot_tag(tag: str) -> torch.Tensor:
    """Transform html tag into one-hot tensor"""
    embedding = torch.zeros((len(TAG_TO_INT.keys()), 1))

    if tag in TAG_TO_INT.keys():
        embedding[TAG_TO_INT[tag]] = 1
    else:
        embedding[TAG_TO_INT["UNK"]] = 1

    return embedding


def one_hot_visibility(visibility: str) -> torch.Tensor:
    """Transform visibility of element into one-hot tensor"""
    embedding = torch.zeros((len(VISIBILITY_TO_INT.keys()), 1))
    embedding[VISIBILITY_TO_INT[visibility]] = 1

    return embedding


def load_pretrained_word_embeddings_tensor(word_first: bool = True) -> Tuple[List[str], Dict[str, int], torch.Tensor]:
    """
    Helper function to load embeddings as tensors and return the tensor representation of the embeddings.

    :return: A tuple containing: Dictionary of word to index, Vocabulary list, Tensor of embeddings.
    """
    embeddings_dict = load_pretrained_word_embeddings()
    embeddings_tensor = torch.Tensor(list(embeddings_dict.values()))
    words = list(embeddings_dict.keys())
    if word_first:
        words = [str(tuple(reversed(ast.literal_eval(word)))) for word in words]
    index_to_word: List[str] = words
    word_to_index: Dict[str, int] = {word: index for index, word in enumerate(index_to_word)}
    del embeddings_dict
    return index_to_word, word_to_index, embeddings_tensor


def load_pretrained_word_embeddings() -> Dict[str, Any]:
    """
    Helper function to load pretrained model into a dictionary of string to vector.

    :return: Dictionary that has (word, country) as keys and vectors as values.
    """
    with open(Path(".word_embeddings") / "embeddings_dict" / "embeddings_dict_pickle", "rb") as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict


def load_fasttext_english_word_embeddings() -> Dict[str, Any]:
    """
    Helper function to load pretrained model into a dictionary of string to vector.

    :return: Dictionary that has word as keys and vectors as values.
    """
    with open(Path(".word_embeddings") / "embeddings_dict" / "english_embeddings_dict.pickle", "rb") as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict


def load_glove_model(embedding_dimension: int) -> Dict[str, Any]:
    """
    Helper function to load glove model into a dictionary of string to vector.

    :param embedding_dimension: The embedding dimension of the model to load.
    :return: Dictionary that has words as keys and vectors as values.
    """
    with open(Path("word_embedding_models") / f"glove.6B.{embedding_dimension}d.txt", "r") as f:
        embeddings_dictionary = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dictionary[word] = vector
    return embeddings_dictionary


def load_fasttext_model(language: str, word_filter: List[str] = None) -> Dict[str, Any]:
    """
    Helper function to load fasttext model into a dictionary of (word, language) tuple to vector.
    Supported languages are ['fi', 'nl', 'sv', 'no', 'en', 'de']

    Set word filter to return embeddings for the words in the list. In this case, the loading will be done in disk
    and the whole model want be loaded to memory.

    :param language: Language to return embeddings for.
    :param word_filter: List of words to return embeddings for.
    :return: Dictionary that has (word, language) as keys and vectors as values.
    """
    vectors_path = Path(".word_embeddings") / f"wiki.{language}.align.vec"

    embeddings_dictionary = {}
    with tqdm(total=os.path.getsize(vectors_path)) as progress_bar:
        with open(vectors_path) as file:
            next(file)  # skip first line with vector information.

            if not word_filter:
                for line in file:
                    progress_bar.update(len(line.encode("utf-8")))
                    values = line.rstrip().rsplit(" ")
                    word = values[0]
                    vector = [float(value) for value in values[1:]]
                    embeddings_dictionary[str((language, word))] = vector
            else:

                filter_dict = Counter(word_filter)
                for line in file:
                    progress_bar.update(len(line.encode("utf-8")))
                    values = line.rstrip().rsplit(" ")
                    word = values[0]
                    if word not in filter_dict:
                        continue
                    vector = [float(value) for value in values[1:]]
                    embeddings_dictionary[str((language, word))] = vector

    return embeddings_dictionary
