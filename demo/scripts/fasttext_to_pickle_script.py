"""
Helper script to convert fasttext embeddings to pickle to be used by the models
"""

import io
import logging
import pickle

import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    logger.info("Pickling fasttext embeddings:")
    fin = io.open('crawl-300d-2M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(value) for value in tokens[1:]]

    with (pathlib.Path(".word_embeddings/embeddings_dict/english_embeddings_dict.pickle")).open(
            mode="wb") as pickle_file:
        pickle.dump(data, pickle_file)


if __name__ == "__main__":
    run()
