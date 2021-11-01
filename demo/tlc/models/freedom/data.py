# pylint: disable=not-callable
"""
Module containing FreeDOM feature classes.
"""
import re
from collections import namedtuple
from typing import Dict, List

import numpy
import torch
from webtraversallibrary import PageElement

from tlc.dataset.singletons import get_character_indexer, get_max_word_length, get_pretrained_fasttext_word_embeddings
from tlc.dataset.utilities import COUNTRY_TO_LANGUAGE, TAG_TO_INT, bounding_box, one_hot_visibility
from tlc.dataset.vocabulary.vocabulary import Vocabulary
from tlc.device import get_torch_device
from tlc.models.freedom.text_embedding import CharacterLevelWordEmbeddingLayer
from tlc.structures.mixins import FeatureMixin

MAX_WORD_COUNT = 15

FreeDOMFeatureTensors = namedtuple(
    "FreeDOMFeatureTensors",
    [
        "feature_text_indexes",
        "feature_character_indexes",
        "feature_tag",
        "feature_node_string_types",
        "feature_location",
        "feature_xpath",
        "feature_class_probabilities",
        "feature_node_embedding",
        "feature_common",
    ],
)


class FreeDOMFeatureMixin(FeatureMixin):
    @classmethod
    def _features(cls, element: PageElement):
        return FreeDOMFeatures(element)


class FreeDOMFeatures:
    """
    Feature Class for FreeDOM to encapsulate the logic of converting a PageElement to tensors.
    """

    SPACY_FEATURES = {
        "like_url",
        "like_email",
        "is_currency",
        "is_title",
        "is_digit",
        "is_lower",
        "is_upper",
        "like_num",
        "is_stop",
        "LOC",
        "MISC",
        "ORG",
        "PER",
        "MONEY",
        "PRODUCT",
        "QUANTITY",
    }  # len = 16

    def __init__(self, element: PageElement = None):
        super().__init__()
        # self.tensor = None
        # if vocabulary_indexer is None:
        #     vocabulary_indexer = get_vocabulary_indexer()
        # self.vocabulary_indexer = vocabulary_indexer

        # if max_word_length is None:
        #     max_word_length = get_max_word_length()
        # self.max_word_length = max_word_length

        # if character_vocabulary_indexer is None:
        #     character_vocabulary_indexer = get_character_indexer()
        # self.character_vocabulary_indexer = character_vocabulary_indexer

        if element is not None:

            pretrained_word_embeddings, embedding_dimension = get_pretrained_fasttext_word_embeddings()

            # Common Features
            self.font_weight = float(element.metadata["font_weight"])  # 1
            self.font_size = float(element.metadata["font_size"].split("px")[0])  # 1
            self.num_imgs = float(element.metadata["num_imgs"])  # 1
            self.num_svgs = float(element.metadata["num_svgs"])  # 1
            self.visibility = one_hot_visibility(element.metadata["visibility"])  # 6
            self.bbox = bounding_box(element.metadata["location"], element.metadata["size"])  # 4

            # Stage 1 Features
            self.feature_market = element.page.page_metadata["ip_location"]["country"]
            self.feature_language = COUNTRY_TO_LANGUAGE[self.feature_market]
            feature_node_text = element.metadata["text_local_clean"] or ""
            self.feature_node_html_tags = element.metadata["tag"]
            self.feature_text_indexes = self.build_english_text_tensor(
                node_text=feature_node_text,
                max_word_length=get_max_word_length(),
                max_word_count=MAX_WORD_COUNT,
                pretrained_word_embeddings=pretrained_word_embeddings,
                embedding_dimension=embedding_dimension,
            )
            self.feature_character_indexes = self.build_text_character_tensor(
                node_text=feature_node_text,
                character_vocabulary_indexer=get_character_indexer(),
                max_word_length=get_max_word_length(),
                max_word_count=MAX_WORD_COUNT,
            )

            # Stage 2 Features
            self.feature_xpath = element.metadata["xpath"][1:].split("/")
            self.feature_text_spacy_features = element.metadata["text_spacy_features"]
            self.feature_location = element.metadata["node_relative_location"]

            self.feature_xpath_indexes = self.build_xpath_tensor(self.feature_xpath, TAG_TO_INT)

            self.feature_tentative_label = []
            self.feature_class_probabilities = []
            self.feature_node_embedding = []

            if "freedom_metadata" in element.metadata:
                self.feature_tentative_label = element.metadata["freedom_metadata"]["tentative_label"]
                self.feature_class_probabilities = element.metadata["freedom_metadata"]["class_probabilities"]
                self.feature_node_embedding = element.metadata["freedom_metadata"]["node_embedding"]

    @property
    def tensors(self) -> FreeDOMFeatureTensors:
        """
        Method retrieving the tensors representation. Computes this representation lazily.
        :return:
        """

        return self.to_tensors()

    def to_tensors(self) -> FreeDOMFeatureTensors:
        """
        Return features as Tensor representations.

        :return: Tuple of tensors.
        """

        feature_text_indexes = torch.tensor(self.feature_text_indexes, device=get_torch_device(), dtype=torch.float)
        feature_character_indexes = torch.tensor(
            self.feature_character_indexes, device=get_torch_device(), dtype=torch.long
        )
        feature_tag = torch.tensor(
            [TAG_TO_INT.get(self.feature_node_html_tags, 0)],
            device=get_torch_device(),  # pylint: disable=no-value-for-parameter
            dtype=torch.long,
        )

        if len(self.feature_text_spacy_features) > 0:
            feature_node_string_types = torch.tensor(
                list(self.feature_text_spacy_features.values()), device=get_torch_device(), dtype=torch.float
            )
        else:
            feature_node_string_types = torch.zeros(
                len(self.SPACY_FEATURES), device=get_torch_device(), dtype=torch.float
            )

        feature_location = torch.tensor(self.feature_location, device=get_torch_device(), dtype=torch.long)
        feature_xpath = torch.tensor(self.feature_xpath_indexes, device=get_torch_device(), dtype=torch.long)

        feature_parts = (
            self.bbox,
            self.font_weight,
            self.font_size,
            self.num_imgs,
            self.num_svgs,
            self.visibility,
            # is_active,
            # tag,
        )

        feature_common = torch.as_tensor(
            numpy.vstack(feature_parts),
            dtype=torch.float,
            device=get_torch_device(),  # pylint: disable=no-value-for-parameter
        ).squeeze()

        return FreeDOMFeatureTensors(
            feature_text_indexes=feature_text_indexes,  # shape: word count
            feature_character_indexes=feature_character_indexes,  # shape: word count * max_word_length
            feature_tag=feature_tag,
            feature_node_string_types=feature_node_string_types,
            feature_location=feature_location,
            feature_xpath=feature_xpath,
            feature_class_probabilities=torch.tensor(
                self.feature_class_probabilities, device=get_torch_device(), dtype=torch.float
            ),
            feature_node_embedding=torch.tensor(
                self.feature_node_embedding, device=get_torch_device(), dtype=torch.float
            ),
            feature_common=feature_common,
        )

    @classmethod
    def build_xpath_tensor(cls, xpath: List[str], tag_to_int: Dict[str, int]) -> List[int]:
        """
        Build a tensor of indices from xpath list

        :param xpath: xpath list.
        :param tag_to_int: Dictionary of html tag to index
        :return: List of indices.
        """
        return [tag_to_int.get(tag, 0) for tag in xpath]

    @classmethod
    def build_english_text_tensor(
        cls,
        node_text: str,
        max_word_length: int,
        max_word_count: int,
        pretrained_word_embeddings: Dict,
        embedding_dimension: int,
    ) -> List:
        """
        Build a tensor of indices for the words in a text.

        Words will be trimmed to fit max_word_length.
        The whole text will be trimmed to fit max_word_count.

        :param node_text: Text string.
        :param max_word_length: Maximum word length allowed.
        :param max_word_count: Maximum number of words allowed in text.
        :param pretrained_word_embeddings: Dictionary of word to embeddings
        :param embedding_dimension: dimension of embeddings in the pretrained embeddings dictionary.

        :return: LongTensor with shape (word count)
        """
        node_words = node_text.split(" ")[:max_word_count]
        return [
            pretrained_word_embeddings.get(text[:max_word_length], numpy.random.rand(embedding_dimension).tolist())
            for text in node_words
        ]

    @classmethod
    def build_multilingual_text_tensor(
        cls,
        language: str,
        node_text: str,
        vocabulary_indexer: Dict[str, int],  # pylint: disable=unused-argument
        max_word_length: int,
        max_word_count: int,
        pretrained_word_embeddings: Dict,
    ) -> List:
        """
        Build a tensor of indices for the words in a text.

        Words will be trimmed to fit max_word_length.
        The whole text will be trimmed to fit max_word_count.

        :param language: Language of the text.
        :param node_text: Text string.
        :param vocabulary_indexer: Dictionary of (language, word) to index.
        :param max_word_length: Maximum word length allowed.
        :param max_word_count: Maximum number of words allowed in text.
        :param pretrained_word_embeddings: Dictionary of word to embeddings

        :return: LongTensor with shape (word count)
        """
        embedding_dimension = len(next(iter(pretrained_word_embeddings.values())))
        node_words = node_text.split(" ")[:max_word_count]
        return [
            pretrained_word_embeddings.get(
                str((language, text[:max_word_length])), numpy.random.rand(embedding_dimension).tolist()
            )
            for text in node_words
        ]

    @classmethod
    def build_text_character_tensor(
        cls, node_text, character_vocabulary_indexer: Dict[str, int], max_word_length: int, max_word_count: int
    ):
        """
        Build a tensor of indices for the words in a text on a character level.

        Words will be trimmed to fit max_word_length also shorter words will be padded to fit max_word_length.
        The whole text will be trimmed to fit max_word_count.

        :param node_text: Text string.
        :param character_vocabulary_indexer: Dictionary of character to index.
        :param max_word_length: Maximum word length allowed.
        :param max_word_count: Maximum number of words allowed in text.
        :return: LongTensor with shape (word count * maximum_word_length)
        """
        node_words = node_text.split(" ")[:max_word_count]
        return [
            cls.build_word_character_tensor(text, character_vocabulary_indexer, max_word_length) for text in node_words
        ]

    @classmethod
    def build_word_tensor(cls, language: str, word: str, vocabulary_indexer: Dict[str, int], max_word_length: int):
        """
        Build index representation of word.

        Words will be trimmed to fit max_word_length.

        :param language: Language of the text.
        :param word: word string.
        :param vocabulary_indexer: Dictionary of (language, word) to index.
        :param max_word_length: Maximum word length allowed.
        :return: Index of the word in the vocabulary.
        """
        return vocabulary_indexer.get(
            str((language, word[:max_word_length])), vocabulary_indexer[Vocabulary.UNKNOWN_TOKEN]
        )

    @classmethod
    def build_word_character_tensor(
        cls, word: str, character_vocabulary_indexer: Dict[str, int], max_word_length: int
    ) -> List[int]:
        """
        Build index representation of word on a character level.

        Words will be trimmed to fit max_word_length also shorter words will be padded to fit max_word_length.

        :param word: word string.
        :param character_vocabulary_indexer: Dictionary of character to index.
        :param max_word_length: Maximum word length allowed.
        :return: List of character indices for the word.
        """

        return CharacterLevelWordEmbeddingLayer.pad_with_constant(
            [
                character_vocabulary_indexer.get(character, character_vocabulary_indexer[Vocabulary.UNKNOWN_TOKEN])
                for character in word[:max_word_length]
            ],
            max_word_length,
            0,
        )

    @classmethod
    def build_string_type_features_dictionary(cls, spacy_tokens) -> Dict[str, bool]:
        """
        Build string type feature vector from spacy tokens.

        :param spacy_tokens: spacy tokens to build feature vector for.
        :return: Feature vector as Tensor
        """
        string_type_features: Dict[str, bool] = {}
        for token in spacy_tokens:  # TODO Remove token.is_url duplicate
            string_type_features["like_url"] = string_type_features.get("like_url", False) or token.like_url
            string_type_features["like_email"] = string_type_features.get("like_email", False) or token.like_email
            string_type_features["is_currency"] = string_type_features.get("is_currency", False) or token.is_currency
            string_type_features["is_title"] = string_type_features.get("is_title", False) or token.is_title
            string_type_features["like_url"] = string_type_features.get("like_url", False) or token.like_url
            string_type_features["is_digit"] = string_type_features.get("is_digit", False) or token.is_digit
            string_type_features["is_lower"] = string_type_features.get("is_lower", False) or token.is_lower
            string_type_features["is_upper"] = string_type_features.get("is_upper", False) or token.is_upper
            string_type_features["like_num"] = string_type_features.get("like_num", False) or token.like_num
            string_type_features["is_stop"] = string_type_features.get("is_stop", False) or token.is_stop
            string_type_features["LOC"] = string_type_features.get("LOC", False) or token.ent_type_ == "LOC"
            string_type_features["MISC"] = string_type_features.get("MISC", False) or token.ent_type_ == "MISC"
            string_type_features["ORG"] = string_type_features.get("ORG", False) or token.ent_type_ == "ORG"
            string_type_features["PER"] = string_type_features.get("PER", False) or token.ent_type_ == "PER"
            string_type_features["MONEY"] = string_type_features.get("MONEY", False) or token.ent_type_ == "MONEY"
            string_type_features["PRODUCT"] = string_type_features.get("PRODUCT", False) or token.ent_type_ == "PRODUCT"
            string_type_features["QUANTITY"] = (
                string_type_features.get("QUANTITY", False) or token.ent_type_ == "QUANTITY"
            )

        return string_type_features

    @classmethod
    def build_string_type_tensor(cls, spacy_tokens) -> torch.Tensor:
        """
        Build string type feature vector from spacy tokens.

        :param spacy_tokens: spacy tokens to build feature vector for.
        :return: Feature vector as Tensor
        """

        string_type_features = [0] * 14
        for token in spacy_tokens:
            if token.like_url:
                string_type_features[0] = 1
            if token.like_email:
                string_type_features[1] = 1
            if token.is_currency:
                string_type_features[2] = 1
            if token.is_title:
                string_type_features[3] = 1
            if token.like_url:
                string_type_features[4] = 1
            if token.is_digit:
                string_type_features[5] = 1
            if token.is_lower:
                string_type_features[6] = 1
            if token.is_upper:
                string_type_features[7] = 1
            if token.like_num:
                string_type_features[8] = 1
            if token.is_stop:
                string_type_features[9] = 1
            if token.ent_type_ == "LOC":
                string_type_features[10] = 1
            if token.ent_type_ == "MISC":
                string_type_features[11] = 1
            if token.ent_type_ == "ORG":
                string_type_features[12] = 1
            if token.ent_type_ == "PER":
                string_type_features[13] = 1

        return torch.tensor(
            string_type_features, device=get_torch_device(), dtype=torch.long  # pylint: disable=no-value-for-parameter
        )


class FreeDOMStageTwoFeatureMixin(FeatureMixin):
    @classmethod
    def _features(cls, element: PageElement):
        return FreeDOMStageTwoFeatures(element)


class FreeDOMStageTwoFeatures(FreeDOMFeatures):
    """
    Feature Class for FreeDOM Stage two to encapsulate the logic of converting a PageElement to tensors.
    """

    def __init__(
        self,
        element: PageElement = None,
    ):
        super().__init__(element)
        if element is not None:
            self.feature_tentative_label = element.metadata["freedom_metadata"]["tentative_label"]
            self.feature_class_probabilities = element.metadata["freedom_metadata"]["class_probabilities"]
            self.feature_node_embedding = element.metadata["freedom_metadata"]["node_embedding"]
            self.feature_position = element.metadata["location"]
            self.feature_size = element.metadata["size"]
            self.feature_xpath: List[str] = element.metadata["path"]

    def to_tensors(self):
        """
        Return features as Tensor representations.

        :return: Tuple of tensors.
        """

        xpath = torch.tensor(
            [
                TAG_TO_INT.get(elem, 0)
                for elem in [re.sub("[\\[][\\d]+?[]]", "", e) for e in self.feature_xpath if e is not None and e != ""]
            ],
            device=get_torch_device(),  # pylint: disable=no-value-for-parameter
            dtype=torch.long,
        )

        location = torch.tensor(
            [
                self.feature_position["x"],
                self.feature_position["y"],
                self.feature_size["width"],
                self.feature_size["height"],
            ],
            device=get_torch_device(),  # pylint: disable=no-value-for-parameter
            dtype=torch.float,
        )

        parent_features = super().to_tensors()
        return (
            *parent_features,
            torch.tensor(self.feature_class_probabilities, device=get_torch_device(), dtype=torch.float),
            torch.tensor(self.feature_node_embedding, device=get_torch_device(), dtype=torch.float),
            xpath,
            location,
        )
