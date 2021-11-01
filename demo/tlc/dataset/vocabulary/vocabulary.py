"""
Module containing Vocabulary class
"""

import logging
from typing import Dict, List

from tlc.dataset.mixins import JsonMixin, PickleMixin


class Vocabulary(JsonMixin, PickleMixin):
    """
    Helper class for creating and using a vocabulary.
    """

    UNKNOWN_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"
    default_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNKNOWN_TOKEN]

    def __init__(self, tokens: List[str], contain_defaults=False, metadata=None):
        if metadata is None:
            metadata = {}
        self.contains_defaults = contain_defaults
        if contain_defaults:
            self._tokens = Vocabulary.default_tokens.copy() + tokens.copy()
        else:
            self._tokens = tokens.copy()
        self.token_to_index = {token: index for index, token in enumerate(self._tokens)}
        self.index_to_token = dict(enumerate(self._tokens))
        self.metadata = metadata

    def get_tokens(self, contain_defaults=False) -> List[str]:
        """
        Returns the list of tokens.
        Set `contain_defaults` to `False` to return original token list without the defaults.

        :param contain_defaults:
        :return: The list of tokens in the vocabulary.
        """
        if not contain_defaults and self.contains_defaults:
            defaults_index = len(Vocabulary.default_tokens)
            return self._tokens[defaults_index:]
        return self._tokens

    def add_token(self, token: str):
        """
        Add token to the vocabulary.

        :param token: token to be added to the vocabulary.
        """
        if token in self.token_to_index:
            return

        self.token_to_index[token] = self.token_count()
        self.index_to_token[self.token_count()] = token
        self._tokens.append(token)

    def add_tokens(self, tokens: List[str]):
        """
        Add a lists of tokens to the vocabulary.

        :param tokens: List of tokens to be added to the vocabulary.
        :return:
        """
        for token in tokens:
            self.add_token(token)

    def token_count(self, contain_defaults=False) -> int:
        """
        Return token count. Set `contain_defaults` to `False` to exclude the defaults from the count.
        return the length of the original tokens excluding the defaults.
        :return:
        """
        length = len(self._tokens)
        if not contain_defaults and self.contains_defaults:
            length -= len(Vocabulary.default_tokens)

        return length

    def to_json(self) -> Dict:
        """
        Method that returns json representation of the vocabulary object.

        :return: Json representation of the vocabulary object
        """
        return {
            "token_count": self.token_count(),
            "tokens": self._tokens,
            "contains_defaults": self.contains_defaults,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, object_json):
        """
        Method to load vocabulary class from a json.

        :param object_json: Json holding the vocabulary information.
        :return: Vocabulary object loaded with the json.
        """
        try:
            vocabulary = Vocabulary(
                tokens=object_json["tokens"],
                contain_defaults=object_json["contains_defaults"],
                metadata=object_json["metadata"],
            )
        except KeyError:
            logging.error("Vocabulary JSON does not have the correct format.")
            raise
        else:
            return vocabulary
