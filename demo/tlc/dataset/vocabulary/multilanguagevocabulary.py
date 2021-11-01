"""
Module containing MultiLanguageVocabulary class.
"""

import logging
from typing import Dict, List, Tuple

from tlc.dataset.mixins import JsonMixin, PickleMixin
from tlc.dataset.vocabulary.vocabulary import Vocabulary


class MultiLanguageVocabulary(JsonMixin, PickleMixin):
    """
    Helper class for creating and using a MultiLanguageVocabulary.
    """

    def __init__(self, tokens_map: Dict[str, Vocabulary] = None, metadata=None):
        if tokens_map is None:
            tokens_map = {}
        if metadata is None:
            metadata = {}
        self._vocabulary_map: Dict[str, Vocabulary] = tokens_map
        self.metadata = metadata

    @property
    def languages(self) -> List[str]:
        return list(self._vocabulary_map.keys())

    @property
    def total_token_count(self) -> int:
        return sum([vocab.token_count() for vocab in self._vocabulary_map.values()])

    @property
    def language_count(self) -> int:
        return len(self.languages)

    def get_language_token_count(self, language: str) -> int:
        """
        Get number of tokens in the specified language.

        :param language:  Language of vocabulary to return token count for.
        :return: Token count of the given language.
        """
        if language in self._vocabulary_map:
            return self._vocabulary_map[language].token_count()
        return 0

    def get_vocabulary(self, language: str) -> Vocabulary:
        """
        Get vocabulary of the specified language.

        :param language: Language of the vocabulary to retrieve.
        :return: Vocabulary object
        """

        return self._vocabulary_map[language]

    def add_vocabulary(self, language: str, vocabulary: Vocabulary):
        """
        Add a language with tokens to the vocabulary.

        :param language: Language to be added to vocabulary.
        :param vocabulary: Tokens of the language to be added.
        """

        assert language not in self._vocabulary_map
        self._vocabulary_map[language] = vocabulary

    def add_token(self, language: str, token: str):
        """
        Add token to specific language. If language doesn't exist it will be added with this single token.

        :param language: Language to add token to.
        :param token: Token to be added to the language.
        """
        if language not in self._vocabulary_map:
            self._vocabulary_map[language] = Vocabulary([])

        self._vocabulary_map[language].add_token(token)

    def add_tokens(self, language: str, tokens: List[str]):
        """
        Add a list of tokens to specific language. If language doesn't exist it will be added with these tokens.

        :param language: Language to add tokens to.
        :param tokens: Tokens to be added to the language.
        """
        if language not in self._vocabulary_map:
            self._vocabulary_map[language] = Vocabulary(tokens)
        else:
            self._vocabulary_map[language].add_tokens(tokens)

    def get_token_tuples(self, contain_defaults=False) -> List[Tuple[str, str]]:
        """
        Get all tokens in the vocabulary as a (language, token) tuple.

        :param contain_defaults: Flag determining if the tokens should contain default tokens.
        :return: A list of (language, token) tuples.
        """
        token_tuples: List[Tuple[str, str]] = []
        for lang, vocabulary in self._vocabulary_map.items():
            token_tuples += zip(
                [lang] * vocabulary.token_count(contain_defaults=contain_defaults),
                vocabulary.get_tokens(contain_defaults=contain_defaults),
            )

        return token_tuples

    def to_json(self) -> Dict:
        """
        Return json representation of the vocabulary map.
        :return: Json representation of class.
        """

        return {
            "language_count": self.language_count,
            "languages": self.languages,
            "token_count": self.total_token_count,
            "vocabularies": {language: vocabulary.to_json() for language, vocabulary in self._vocabulary_map.items()},
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
            multi_language_vocabulary = MultiLanguageVocabulary(
                tokens_map={
                    language: Vocabulary.from_json(language_json)
                    for language, language_json in object_json["vocabularies"].items()
                },
                metadata=object_json["metadata"],
            )
        except KeyError:
            logging.error("Vocabulary JSON does not have the correct format.")
            raise
        else:
            return multi_language_vocabulary
