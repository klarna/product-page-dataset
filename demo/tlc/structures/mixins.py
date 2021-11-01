"""
Mixin classes used to customize the feature and label encodings for trees in the dataset
"""
from typing import Tuple

import numpy as np
import torch
import webtraversallibrary as wtl
from torch import Tensor

from ..dataset.singletons import get_universal_sentence_encoder
from ..dataset.utilities import bounding_box, one_hot_tag, one_hot_visibility, pre_process
from ..device import get_torch_device


class FeatureMixin:
    """
    Mixin class that defines the basic behavior for feature transformation.
    Needs to implement the `_features` method. Overriding mixins need to inherit from this
    class to ensure correct MRO.
    """

    @staticmethod
    def _features(element: wtl.PageElement) -> Tensor:
        """
        Extract the features from a labeled page element
        """
        tag = one_hot_tag(element.metadata["tag"])
        font_weight = float(element.metadata["font_weight"])
        font_size = float(element.metadata["font_size"].split("px")[0])
        num_imgs = float(element.metadata["num_imgs"])
        num_svgs = float(element.metadata["num_svgs"])
        visibility = one_hot_visibility(element.metadata["visibility"])

        bbox = bounding_box(element.metadata["location"], element.metadata["size"])

        feature_parts = (
            bbox,
            font_weight,
            font_size,
            num_imgs,
            num_svgs,
            visibility,
            # is_active,
            tag,
        )

        return torch.as_tensor(
            np.vstack(feature_parts),
            dtype=torch.float64,
            device=get_torch_device(),  # pylint: disable=no-value-for-parameter
        ).squeeze()


class LabelMixin:
    """
    Mixin class that defines the basic behavior for label transformation.
    Needs to implement the `_label` method. Overriding mixins need to inherit from this
    class to ensure correct MRO.
    """

    DEFAULT_LABEL = "unlabeled"
    HARD_LABEL = "klarna-ai-hard"

    DATAPOINT_LABEL_COLUMN = "klarna-ai-label"
    NODE_LABELS = ["subjectnode", "cart", "name", "price", "addtocart", "mainpicture"]
    ALL_LABELS = [DEFAULT_LABEL] + NODE_LABELS

    @classmethod
    def _label(cls, element: wtl.PageElement) -> str:
        """
        Extract the label from a labeled page element
        """
        if cls.DATAPOINT_LABEL_COLUMN in element.metadata["attributes"]:
            _label = element.metadata["attributes"][cls.DATAPOINT_LABEL_COLUMN].lower().replace(" ", "")
        else:
            _label = cls.DEFAULT_LABEL

        return _label

    @classmethod
    def label_encode(cls, label: str) -> int:
        return cls.ALL_LABELS.index(label)

    @classmethod
    def label_decode(cls, index: int) -> str:
        return cls.ALL_LABELS[index]


class PairLabelMixin(LabelMixin):
    """
    Mixin class for pair labels used by FreeDOM.
    """

    VALUE_LABEL = "value"
    NONE_LABEL = "none"
    ALL_LABELS = ["none-none", "value-none", "none-value", "value-value"]

    @classmethod
    def _label(cls, element: Tuple[wtl.PageElement, wtl.PageElement]) -> str:
        """
        Extract the label from a pair of page elements.

        :param head_element: Head page element.
        :param tail_element: Tail page element.
        :return: Label of the pair of elements.
        """
        head_element, tail_element = element

        if cls.DATAPOINT_LABEL_COLUMN in head_element.metadata["attributes"]:
            head_label = cls.VALUE_LABEL
        else:
            head_label = cls.NONE_LABEL

        if cls.DATAPOINT_LABEL_COLUMN in tail_element.metadata["attributes"]:
            tail_label = cls.VALUE_LABEL
        else:
            tail_label = cls.NONE_LABEL

        return f"{head_label}-{tail_label}"

    @classmethod
    def split_label(cls, label: str) -> Tuple[str, str]:
        label_splitted = label.split("-")
        return label_splitted[0], label_splitted[1]


class ElementMixin(FeatureMixin):
    """Mixin class for accessing the element of a node"""

    @staticmethod
    def _features(element: wtl.PageElement):
        return element


class UniversalSentenceFeatureMixin(FeatureMixin):
    """
    Mixin class that defines the behavior for local text feature transformation based on the Universal Sentence Embedder
    """

    @staticmethod
    def _features(element, sentence_dim: int = 512):  # pylint: disable = W0221
        """
        Extract the features from a labeled page element
        """
        text = element.metadata["text_local"]
        text = pre_process(text)

        universal_embedder = get_universal_sentence_encoder()

        # Embed the node's text if it is not an empty string

        if len(text) > 1 and text != "" and text != " ":
            text_list = text.split()

            # Limit the number of words
            text_list = text_list[:100]
            # Limit the character length in each word
            text_list = [text[:20] for text in text_list]
            text = " ".join(text_list)

            with torch.no_grad():
                text_embedding = torch.from_numpy(universal_embedder([text]).numpy())
        else:
            # Zero padding if the node contains no text.
            text_embedding = torch.zeros(sentence_dim)

        # The html markup features
        tag = one_hot_tag(element.metadata["tag"])
        font_weight = float(element.metadata["font_weight"])
        font_size = float(element.metadata["font_size"].split("px")[0])
        num_imgs = float(element.metadata["num_imgs"])
        num_svgs = float(element.metadata["num_svgs"])
        visibility = one_hot_visibility(element.metadata["visibility"])
        bbox = bounding_box(element.metadata["location"], element.metadata["size"])
        feature_parts = (
            bbox,
            font_weight,
            font_size,
            num_imgs,
            num_svgs,
            visibility,
            tag,
            text_embedding.unsqueeze(-1).squeeze(0),
        )

        return torch.as_tensor(np.vstack(feature_parts), dtype=torch.float64).squeeze()
