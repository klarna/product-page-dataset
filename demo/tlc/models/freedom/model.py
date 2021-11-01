"""
Module containing freeDOM <https://arxiv.org/abs/2010.10755> implementation.
"""
from dataclasses import dataclass
from typing import Any, Dict, List

import gin
import torch
import torch.nn.functional
from torch import nn

from tlc.models.freedom.text_embedding import TextEmbeddingModule
from tlc.models.prototypes import TreeEmbedder
from tlc.structures.trees import FreeDOMDataTree


class LocalNodeRepresentationModule(nn.Module):
    """
    Class for the first stage representation of the freeDOM model.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        character_embedding_dim,
        word_embedding_dim,
        cnn_character_embedding_kernel_size,
        cnn_character_embedding_filter_count,
        rnn_hidden_dimension,
        word_vocabulary: List[str],
        max_word_length: int,
        character_vocabulary: List[str],
        pretrained_word_embedding_model: Dict[str, Any],
        markup_features_count,
        string_type_features_count,
        discrete_features_embedding_dim,
        common_features_embedding_dim,
    ):
        super().__init__()

        self.markup_features_count = markup_features_count
        self.string_type_features_count = string_type_features_count
        self.discrete_features_embedding_dim = discrete_features_embedding_dim
        self.common_features_embedding_dim = common_features_embedding_dim
        self.text_embedding_module = TextEmbeddingModule(
            character_embedding_dim=character_embedding_dim,
            word_embedding_dim=word_embedding_dim,
            cnn_character_embedding_kernel_size=cnn_character_embedding_kernel_size,
            cnn_character_embedding_filter_count=cnn_character_embedding_filter_count,
            rnn_hidden_dimension=rnn_hidden_dimension,
            word_vocabulary=word_vocabulary,
            max_word_length=max_word_length,
            character_vocabulary=character_vocabulary,
            pretrained_word_embedding_model=pretrained_word_embedding_model,
        )
        self.markup_features_embedding_layer = torch.nn.Embedding(
            self.markup_features_count, self.discrete_features_embedding_dim
        )

        self.string_type_embedding_layer = torch.nn.Linear(
            self.string_type_features_count, self.discrete_features_embedding_dim
        )

        self.output_dimension = (
            self.text_embedding_module.output_dimension
            + self.text_embedding_module.output_dimension
            + self.discrete_features_embedding_dim
            + self.discrete_features_embedding_dim
            + self.common_features_embedding_dim
        )

    def forward(self, data_tree):
        """
        Forward pass of model.

        :param data_tree: Instance of freeDOMDataTree.
        :return: Output of forward pass.
        """

        feature_vector_tensor = data_tree.feature_vector.tensors
        text_representation = self.text_embedding_module(
            feature_vector_tensor.feature_text_indexes, feature_vector_tensor.feature_character_indexes
        )

        if data_tree.parent:
            parent_feature_vector_tensor = data_tree.parent.feature_vector.tensors
            feature_parent_text_indexes = parent_feature_vector_tensor.feature_text_indexes
            feature_parent_character_indexes = parent_feature_vector_tensor.feature_character_indexes
        else:
            feature_parent_text_indexes = torch.zeros_like(
                feature_vector_tensor.feature_text_indexes[0],
                device=feature_vector_tensor.feature_text_indexes[0].device,
                dtype=feature_vector_tensor.feature_text_indexes[0].dtype,
            ).unsqueeze(dim=0)
            feature_parent_character_indexes = torch.zeros_like(
                feature_vector_tensor.feature_character_indexes[0],
                device=feature_vector_tensor.feature_character_indexes[0].device,
                dtype=feature_vector_tensor.feature_character_indexes[0].dtype,
            ).unsqueeze(dim=0)

        previous_text_representation = self.text_embedding_module(
            feature_parent_text_indexes, feature_parent_character_indexes
        )

        markup_features = torch.sum(self.markup_features_embedding_layer(feature_vector_tensor.feature_tag), dim=0)
        string_type_features = self.string_type_embedding_layer(feature_vector_tensor.feature_node_string_types)
        discrete_features_representation = torch.cat([markup_features, string_type_features])

        return torch.cat(
            [
                torch.squeeze(text_representation),
                torch.squeeze(previous_text_representation),
                discrete_features_representation,
                feature_vector_tensor.feature_common,
            ]
        )

    def forward_batch(self, batch):
        """
        Forward pass of model.

        :param batch: Batch of freeDOMDataTree.
        :return: Output of forward pass.
        """

        node_text_batch = []
        node_text_char_batch = []

        prev_node_text_batch = []
        prev_node_text_char_batch = []

        html_tag_batch = []
        node_string_types_batch = []

        common_features_batch = []

        for node in batch:

            node_feature_tensors = node.feature_vector.tensors

            if node.parent:
                parent_feature_vector_tensor = node.parent.feature_vector.tensors
                feature_parent_text_indexes = parent_feature_vector_tensor.feature_text_indexes
                feature_parent_character_indexes = parent_feature_vector_tensor.feature_character_indexes
            else:
                feature_parent_text_indexes = torch.zeros_like(
                    node_feature_tensors.feature_text_indexes[0],
                    device=node_feature_tensors.feature_text_indexes[0].device,
                    dtype=node_feature_tensors.feature_text_indexes[0].dtype,
                ).unsqueeze(dim=0)
                feature_parent_character_indexes = torch.zeros_like(
                    node_feature_tensors.feature_character_indexes[0],
                    device=node_feature_tensors.feature_character_indexes[0].device,
                    dtype=node_feature_tensors.feature_character_indexes[0].dtype,
                ).unsqueeze(dim=0)

            node_text_batch.append(node_feature_tensors.feature_text_indexes)
            node_text_char_batch.append(node_feature_tensors.feature_character_indexes)

            prev_node_text_batch.append(feature_parent_text_indexes)
            prev_node_text_char_batch.append(feature_parent_character_indexes)

            html_tag_batch.append(node_feature_tensors.feature_tag)
            node_string_types_batch.append(node_feature_tensors.feature_node_string_types)

            common_features_batch.append(node_feature_tensors.feature_common)

        html_tag_batch = torch.stack(html_tag_batch)
        node_string_types_batch = torch.stack(node_string_types_batch)
        common_features_batch = torch.stack(common_features_batch)

        text_batch_representation = self.text_embedding_module.forward_batch(node_text_batch, node_text_char_batch)
        prev_text_batch_representation = self.text_embedding_module.forward_batch(
            prev_node_text_batch, prev_node_text_char_batch
        )

        markup_features = torch.squeeze(torch.sum(self.markup_features_embedding_layer(html_tag_batch), dim=1))

        string_type_features = self.string_type_embedding_layer(node_string_types_batch)

        discrete_features_representation = torch.cat([markup_features, string_type_features], dim=1)
        return torch.cat(
            [
                torch.squeeze(text_batch_representation),
                torch.squeeze(prev_text_batch_representation),
                discrete_features_representation,
                common_features_batch,
            ],
            dim=1,
        )


class FeedForwardLayer(torch.nn.Module):
    """
    Simple feedforward layer.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of model.

        :param x: Input of layer.
        :return: Output of forward pass.
        """
        return self.sigmoid(self.fc1(x))


@gin.configurable
@dataclass
class LocalModuleHyperParameters:
    """
    Data class for encapsulating local module hyper parameters
    """

    character_embedding_dim: int
    word_embedding_dim: int
    cnn_character_embedding_kernel_size: int
    cnn_character_embedding_filter_count: int
    rnn_hidden_dimension: int
    discrete_features_embedding_dim: int
    common_features_embedding_dim: int
    dropout_rate: float


@gin.configurable
@dataclass
class RelationalModuleHyperParameters:
    """
    Data class for encapsulating local module hyper parameters
    """

    xpath_embedding_dim: int
    position_embedding_dim: int
    xpath_lstm_hidden_dim: int
    position_size: int
    m_top_uncertain_field_nodes: int
    label_voting_threshold: int
    local_node_embedding_model_path: str
    dropout_rate: float


class FreeDOMModule(TreeEmbedder):
    """Class for freeDOM model"""

    def __init__(
        self,
        word_vocabulary: List[str],
        max_word_length: int,
        markup_features_count: int,
        string_type_features_count: int,
        character_vocabulary: List[str],
        pretrained_word_embedding_model: Dict[str, Any],
        feedforward_layer_hidden_dim: int,
        local_module_hyper_parameters: LocalModuleHyperParameters,
    ):
        super().__init__()

        self.local_node_representation_layer = LocalNodeRepresentationModule(
            character_embedding_dim=local_module_hyper_parameters.character_embedding_dim,
            word_embedding_dim=local_module_hyper_parameters.word_embedding_dim,
            cnn_character_embedding_kernel_size=local_module_hyper_parameters.cnn_character_embedding_kernel_size,
            cnn_character_embedding_filter_count=local_module_hyper_parameters.cnn_character_embedding_filter_count,
            rnn_hidden_dimension=local_module_hyper_parameters.rnn_hidden_dimension,
            word_vocabulary=word_vocabulary,
            max_word_length=max_word_length,
            character_vocabulary=character_vocabulary,
            pretrained_word_embedding_model=pretrained_word_embedding_model,
            markup_features_count=markup_features_count,
            string_type_features_count=string_type_features_count,
            discrete_features_embedding_dim=local_module_hyper_parameters.discrete_features_embedding_dim,
            common_features_embedding_dim=local_module_hyper_parameters.common_features_embedding_dim,
        )

        self.local_node_mlp = FeedForwardLayer(
            self.local_node_representation_layer.output_dimension,
            feedforward_layer_hidden_dim,
        )

        self.dropout_layer = torch.nn.Dropout(p=local_module_hyper_parameters.dropout_rate)

        self.output_dimension = feedforward_layer_hidden_dim

    @property
    def node_embedding_dimension(self):
        return self.local_node_representation_layer.output_dimension

    def node_embedding(self, node: FreeDOMDataTree):
        """
        Get local embedding of a node.

        :param node: Instance of freeDOMDataTree.
        :return: Pure embedding of node before passing through MLP.
        """
        return self.local_node_representation_layer(node)

    def node_embedding_batch(self, batch: List[FreeDOMDataTree]):
        """
        Get local embedding of a batch of nodes.

        :param batch: List of freeDOMDataTree's.
        :return: Pure embedding of nodes before passing through MLP.
        """
        return self.local_node_representation_layer.forward_batch(batch)

    def forward(self, node: FreeDOMDataTree):
        """
        Forward pass of model.

        :param node: Instance of freeDOMDataTree.
        :return: Output of forward pass.
        """
        local_node_representation = self.node_embedding(node)
        return self.dropout_layer(self.local_node_mlp(local_node_representation))

    def forward_batch(self, batch):  # pylint: disable=arguments-differ
        """
        Forward pass of model.

        :param batch: Batch of freeDOMDataTree nodes.
        :return: Output of forward pass.
        """
        local_node_batch_representations = self.node_embedding_batch(batch)
        return self.dropout_layer(self.local_node_mlp(local_node_batch_representations))

    def reset(self):
        pass
