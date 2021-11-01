"""
Entrypoint for vocabulary extraction script.
"""
import json
import logging
import os
import pathlib
import pickle

import click

from tlc.cli.decorators import add_options
from tlc.dataset.utilities import load_fasttext_model
from tlc.dataset.vocabulary.multilanguagevocabulary import MultiLanguageVocabulary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

common_options = [
    click.option(
        "--vocabulary-path",
        type=pathlib.Path,
        required=True,
        help="Path of vocabulary to convert.",
    ),
    click.option(
        "--output-embeddings-path",
        type=pathlib.Path,
        required=True,
        help="File path to save embeddings",
    ),
    click.option("--should-pickle", is_flag=True, help="Flag determining if the embeddings should be pickled."),
]


@add_options(common_options)
@click.command()
def run(vocabulary_path: pathlib.Path, output_embeddings_path: pathlib.Path, should_pickle: bool):
    """
    Convert vocabulary to word embeddings using FastText multi language embeddings.
    """
    logger.info("Starting to extract vocabulary embedding matrix.")
    logger.info(f"{should_pickle}")

    os.mkdir(str(output_embeddings_path))
    logger.info("Creating destination folder %s", output_embeddings_path)

    multi_lang_vocab = MultiLanguageVocabulary.from_json_file(vocabulary_path)

    model_dict = {}

    for language in multi_lang_vocab.languages:
        logger.info(f"Creating embeddings dictionary for language {language}")
        model_dict.update(load_fasttext_model(language, multi_lang_vocab.get_vocabulary(language).get_tokens()))
        logger.info(f"Number of embeddings after adding language {language}: {len(model_dict)}")

    with (output_embeddings_path / "embeddings_dict.json").open(mode="w") as json_file:
        json.dump(model_dict, json_file)

    if should_pickle:
        with (output_embeddings_path / "embeddings_dict_pickle").open(mode="wb") as pickle_file:
            pickle.dump(model_dict, pickle_file)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
