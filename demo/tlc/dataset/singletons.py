"""
Module containing singletons for loading vocabularies and large NLP dependencies.
"""

#  pylint: # pylint: disable=global-statement
#  TODO: Find better structure for these objects.

import logging

import spacy
import tensorflow_hub as hub
from pathlib import Path

from tlc.dataset.utilities import load_fasttext_english_word_embeddings, load_pretrained_word_embeddings
from tlc.dataset.vocabulary.multilanguagevocabulary import MultiLanguageVocabulary
from tlc.dataset.vocabulary.vocabulary import Vocabulary

spacy_model = None
vocabulary = None
vocabulary_tuples = None
character_vocabulary = None
character_to_int = None
vocabulary_to_int = None
max_word_length = None
embeddings_dict = None
embedding_dimension = None
pretrained_word_embeddings = None
universal_sentence_encoder = None


def get_spacy_model():
    """
    Helper method to load Spacy model.

    :return: Instance of Spacy model.
    """
    global spacy_model
    if spacy_model is None:
        try:
            spacy_model = spacy.load("en_core_web_sm")
        except IOError:
            logging.error("Spacy model not installed!")
            logging.error("Run `python -m spacy download en_core_web_sm` and then re-run everything.")
            raise
    return spacy_model


def _get_character_vocabulary():
    """
    Helper method to load character vocabulary.

    :return: List of characters in vocabulary.
    """
    global character_vocabulary
    if character_vocabulary is None:
        vocabulary_file_path = Path("vocabularies/dataset_character_vocabulary")
        try:
            character_vocabulary = [Vocabulary.PAD_TOKEN, Vocabulary.UNKNOWN_TOKEN]
            character_vocabulary += Vocabulary.from_pickle_file(vocabulary_file_path).get_tokens()
        except FileNotFoundError:
            logging.error(f"Vocabulary json not found. Copy it to {str(vocabulary_file_path)}.")
    return character_vocabulary


MAX_WORD_LENGTH_ALLOWED = 13


def get_max_word_length():
    """
    Helper method to load max word length in word vocabulary.

    :return: Maximum word length in the vocabulary.
    """
    global max_word_length
    if max_word_length is None:
        pretrained_embedder, _ = get_pretrained_fasttext_word_embeddings()
        max_word_length = min(
            len(max([word for word in pretrained_embedder.keys()], key=len)),
            MAX_WORD_LENGTH_ALLOWED
        )
    return max_word_length


def _get_vocabulary():
    """
    Helper method to load word vocabulary.

    :return: MultiLanguageWordVocabulary object.
    """
    global vocabulary
    if vocabulary is None:
        vocabulary_file_path = Path("vocabularies/dataset_vocabulary")
        try:
            vocabulary = MultiLanguageVocabulary.from_pickle_file(vocabulary_file_path)
        except FileNotFoundError:
            logging.error(f"Vocabulary json not found. Copy it to {str(vocabulary_file_path)}.")
    return vocabulary


def _get_vocabulary_tuples():
    """
    Helper method to load word vocabulary as (language, word) tuples.

    :return: List of (language, word) tuples.
    """
    global vocabulary_tuples
    if vocabulary_tuples is None:
        vocabulary_dict = _get_vocabulary()
        vocabulary_tuples = [Vocabulary.UNKNOWN_TOKEN]
        vocabulary_tuples += vocabulary_dict.get_token_tuples()
    return vocabulary_tuples


def get_character_indexer():
    """
    Helper method to get character to index dictionary.

    :return: Character to index dictionary.
    """
    global character_to_int
    if character_to_int is None:
        character_to_int = dict((item, index) for index, item in enumerate(_get_character_vocabulary()))
    return character_to_int


def get_vocabulary_indexer():
    """
    Helper method to get word to index dictionary.

    :return: Word to index dictionary.
    """
    global vocabulary_to_int
    if vocabulary_to_int is None:
        vocabulary_to_int = dict((str(item), index) for index, item in enumerate(_get_vocabulary_tuples()))
    return vocabulary_to_int


def get_pretrained_word_embeddings():
    """
    Helper function to load pretrained model into a dictionary of string to vector as a singleton

    :return: Dictionary that has (word, country) as keys and vectors as values.
    """
    global pretrained_word_embeddings
    global embedding_dimension
    if pretrained_word_embeddings is None:
        pretrained_word_embeddings = load_pretrained_word_embeddings()
        embedding_dimension = len(next(iter(pretrained_word_embeddings.values())))
    return pretrained_word_embeddings, embedding_dimension


def get_pretrained_fasttext_word_embeddings():
    """
    Helper function to load pretrained model into a dictionary of string to vector as a singleton

    :return: Dictionary that has word as keys and vectors as values.
    """
    global pretrained_word_embeddings
    global embedding_dimension
    if pretrained_word_embeddings is None:
        pretrained_word_embeddings = load_fasttext_english_word_embeddings()
        embedding_dimension = len(next(iter(pretrained_word_embeddings.values())))
    return pretrained_word_embeddings, embedding_dimension


def get_universal_sentence_encoder(pre_trained_url_id=0):
    """pre_trained_url_id = 0 or 1.
    0: standard universal sentence encoder,
    1:  Model trained with a Transformer encoder.

    Loads or creates a pretrained sentence embedder called the universal sentence encoder.
    https://tfhub.dev/google/universal-sentence-encoder.
    The universal-sentence-encoder-large model is trained with a Transformer encoder."""

    pre_trained_urls = [
        "https://tfhub.dev/google/universal-sentence-encoder/4",
        "https://tfhub.dev/google/universal-sentence-encoder-large/5",
    ]
    pre_trained_url = pre_trained_urls[pre_trained_url_id]

    global universal_sentence_encoder

    if universal_sentence_encoder is None:
        print("Loading pre-trained embedder from: %s " % pre_trained_url)
        universal_sentence_encoder = hub.load(pre_trained_url)
        print("Pre-trained embedder loaded")

    return universal_sentence_encoder
