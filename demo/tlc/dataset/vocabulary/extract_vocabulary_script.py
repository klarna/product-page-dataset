"""
Entrypoint for vocabulary extraction script.
"""
import glob
import json
import logging
import os
import pathlib

import bs4
import click
import webtraversallibrary

from tlc.cli.decorators import add_options
from tlc.dataset.utilities import COUNTRY_TO_LANGUAGE, pre_process
from tlc.dataset.vocabulary.multilanguagevocabulary import MultiLanguageVocabulary
from tlc.dataset.vocabulary.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

common_options = [
    click.option(
        "--dataset-path",
        type=pathlib.Path,
        required=True,
        help="Path of dataset directory.",
    ),
    click.option(
        "--output-json",
        type=pathlib.Path,
        required=True,
        help="Local directory to save the vocabulary.",
    ),
    click.option(
        "--max-files",
        type=int,
        required=False,
        help="File limit to create vocabulary for.",
    ),
]


def get_snapshot_from_local_dir(path: pathlib.Path):
    """
    Load WTL snapshot from a local directory.
    :param path: Path to the files of the webpage.
    :return: WTL page snapshot.
    """

    if not (path / "elements_metadata.json").exists():
        return None

    with open(path / "elements_metadata.json") as f:
        element_metadata = json.load(f)
    with open(path / "page_metadata.json") as f:
        page_metadata = json.load(f)
    with open(path / "source.html") as f:
        bs4object = bs4.BeautifulSoup(f, "html5lib")

    return webtraversallibrary.PageSnapshot(bs4object, page_metadata, element_metadata)


@add_options(common_options)
@click.command()
def run(dataset_path: pathlib.Path, output_json: pathlib.Path, max_files: int):
    """
    Creates a vocabulary based on the text in the local node.
    """
    logger.info("Starting to extract vocabulary.")

    if max_files:
        file_paths = glob.glob(str(dataset_path / "*/*/"))[:max_files]
    else:
        file_paths = glob.glob(str(dataset_path / "*/*/"))

    os.mkdir(str(output_json))
    logger.info("Creating destination folder %s", output_json)

    multi_lang_vocab_local = MultiLanguageVocabulary()
    hostname_vocab = Vocabulary([])
    character_level_vocab = Vocabulary([])

    for file_path in file_paths:

        logger.info("Processing page in: " + str(file_path))

        page_snapshot = get_snapshot_from_local_dir(pathlib.Path(file_path))

        country = (
            page_snapshot.page_metadata["ip_location"]["country"]
            if "country" in page_snapshot.page_metadata["ip_location"]
            else "NA"
        )

        hostname_vocab.add_token(page_snapshot.page_metadata["hostname"])
        for element in page_snapshot.elements_metadata:
            local_text = element["text_local"]
            for word in pre_process(local_text).split():
                if len(word) > 1:
                    multi_lang_vocab_local.add_token(COUNTRY_TO_LANGUAGE[country], word)

                for character in word:
                    character_level_vocab.add_token(character)

    hostname_vocab.to_json_file(output_json / "dataset_hostname_vocabulary.json")

    multi_lang_vocab_local.to_json_file(output_json / "dataset_vocabulary.json")

    character_level_vocab.to_json_file(output_json / "dataset_character_vocabulary.json")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
