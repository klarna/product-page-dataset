"""Module containing the definitions of all models"""
from pathlib import Path
from typing import Any, Dict, List, Union

import gin
import torch
from torch import Tensor, nn

from ..dataset.singletons import get_character_indexer, get_max_word_length, get_pretrained_fasttext_word_embeddings, \
    get_vocabulary_indexer
from ..dataset.utilities import TAGS_OF_INTEREST
from ..device import get_torch_device
from ..models.riser.data_utils import path_component_vocab, word_vocab
from ..models.riser.riser import RiSER
from ..structures.trees import DataTree
from .domqnet import DOMQNETModule, DOMQNETWithGlobalEmbeddingModule
from .freedom.data import FreeDOMFeatures
from .freedom.model import FreeDOMModule, LocalModuleHyperParameters, RelationalModuleHyperParameters
from .freedom.node_distance_embedding_model import NodeDistanceEmbeddingModule
from .gat import GATDotProductModule, GATModule
from .gcn import FeedforwardMultiLayerGCNModule, GCNModule, MultiLayerGCNModule
from .gln import GLNModule
from .prototypes import (
    BidirectionalModel,
    BidirectionalModelWithEmbeddings,
    TreeClassifier,
    TreeEmbedder,
    TreePairClassifier,
)
from .transformer import TransformerEncoderModule
from .treelstm import BottomUpTreeLSTM, TopDownTreeLSTM
from .treernn import BottomUpTreeRNN, TopDownTreeRNN


class BottomUpLSTMClassifier(TreeClassifier):
    """Bottom-up LSTM model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        super().__init__(
            BottomUpTreeLSTM(input_dimension, latent_dimension),
            input_dimension,
            output_dimension,
            latent_dimension,
        )


class TopDownLSTMClassifier(TreeClassifier):
    """Top-down LSTM model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        super().__init__(
            TopDownTreeLSTM(input_dimension, latent_dimension),
            input_dimension,
            output_dimension,
            latent_dimension,
        )


class BidirectionalLSTMClassifier(TreeClassifier):
    """Bidirectional LSTM model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        tree_model = BidirectionalModel((BottomUpTreeLSTM, TopDownTreeLSTM), input_dimension, latent_dimension)
        super().__init__(tree_model, input_dimension, output_dimension, 2 * latent_dimension)


class BidirectionalLSTMClassifierWithEmbeddings(TreeClassifier):
    """Bidirectional LSTM model with embeddings for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        tree_model = BidirectionalModelWithEmbeddings(
            (BottomUpTreeLSTM, TopDownTreeLSTM), input_dimension, latent_dimension
        )
        super().__init__(tree_model, input_dimension, output_dimension, 2 * latent_dimension)


class BidirectionalRNNClassifier(TreeClassifier):
    """Bidirectional RNN model for tree structured data"""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        latent_dimension: int,
    ):
        tree_model = BidirectionalModel((BottomUpTreeRNN, TopDownTreeRNN), input_dimension, latent_dimension)
        super().__init__(tree_model, input_dimension, output_dimension, 2 * latent_dimension)


class BidirectionalRNNClassifierWithEmbeddings(TreeClassifier):
    """Bidirectional RNN model with embeddings for tree structured data"""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        latent_dimension: int,
    ):
        tree_model = BidirectionalModelWithEmbeddings(
            (BottomUpTreeRNN, TopDownTreeRNN), input_dimension, latent_dimension
        )
        super().__init__(tree_model, input_dimension, output_dimension, 2 * latent_dimension)


class GATClassifier(TreeClassifier):
    """Multi-headed graph attention model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        super().__init__(
            GATModule(input_dimension, latent_dimension),
            input_dimension,
            output_dimension,
            latent_dimension,
        )


class GATDotProductClassifier(TreeClassifier):
    """Multi-headed graph attention model for tree structured data with dot-product attention"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        super().__init__(
            GATDotProductModule(input_dimension, latent_dimension), input_dimension, output_dimension, latent_dimension
        )


class GCNClassifier(TreeClassifier):
    """Graph convolutional model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        super().__init__(
            GCNModule(input_dimension, latent_dimension),
            input_dimension,
            output_dimension,
            latent_dimension,
        )


class MultiLayerGCNClassifier(TreeClassifier):
    """Graph convolutional model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: Union[int, List[int]]):

        if isinstance(latent_dimension, int):
            latent_dimension = [latent_dimension]

        super().__init__(
            MultiLayerGCNModule(input_dimension, latent_dimension),
            input_dimension,
            output_dimension,
            latent_dimension[-1],
        )


class FeedforwardMultiLayerGCNClassifier(TreeClassifier):
    """Graph convolutional model for tree structured data with local feature feedforward"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: Union[int, List[int]]):

        if isinstance(latent_dimension, int):
            latent_dimension = [latent_dimension]

        super().__init__(
            FeedforwardMultiLayerGCNModule(input_dimension, latent_dimension),
            input_dimension,
            output_dimension,
            latent_dimension[-1],
        )


class GLNClassifier(TreeClassifier):
    """Gated linear network model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        # pylint: disable=non-parent-init-called,super-init-not-called

        nn.Module.__init__(self)
        self.one_vs_all_models = nn.ModuleList()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.latent_dimension = latent_dimension
        self.softmax = nn.Softmax(dim=0)

        for _ in range(output_dimension):
            bin_model = GLNModule(input_dimension, output_dimension=1, layer_dimensions=[74], context_dimension=10)
            self.one_vs_all_models.append(bin_model)

    def forward(self, node: DataTree) -> Tensor:
        input_data = node.feature_vector.float()
        output_data = torch.tensor([], device=get_torch_device())  # pylint: disable=not-callable
        for bin_model in self.one_vs_all_models:
            output_data = torch.cat((output_data, bin_model(input_data)), dim=0)
        return output_data

    def reset(self) -> None:
        pass


class FullyConnectedClassifier(TreeClassifier):
    """Fully connected model for tree structured data"""

    class _Embedder(TreeEmbedder):
        def __init__(self, input_dimension: int, output_dimension: int):
            super().__init__()
            self.layer = nn.Linear(input_dimension, output_dimension)
            self.relu = nn.ReLU()

        def forward(self, node: DataTree) -> torch.Tensor:
            return self.relu(self.layer(node.feature_vector.float()))

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        super().__init__(
            self._Embedder(input_dimension, latent_dimension), input_dimension, output_dimension, latent_dimension
        )


class DOMQNETClassifier(TreeClassifier):
    """Dom-Q-Net model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        super().__init__(
            DOMQNETModule(input_dimension, latent_dimension),
            input_dimension,
            output_dimension,
            latent_dimension,
        )


class DOMQNETWithGlobalEmbeddingClassifier(TreeClassifier):
    """Dom-Q-Net model for tree structured data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        super().__init__(
            DOMQNETWithGlobalEmbeddingModule(input_dimension, latent_dimension),
            input_dimension,
            output_dimension,
            latent_dimension,
        )


class RiSERClassifier(TreeClassifier):
    """Riser model for tree structured data"""

    def __init__(
        self,
        input_dimension: int = 0,
        output_dimension: int = 6,
        latent_dim: int = 150,
    ):

        word_embed_dim = 100
        word_lstm_dim = latent_dim
        word_dropout = 0.25

        xpath_embed_dim = 25
        xpath_lstm_dim = 16
        xpath_dropout = 0.5

        model = RiSER(
            word_embed_dim=word_embed_dim,
            xpath_embed_dim=xpath_embed_dim,
            word_vocab=word_vocab,
            xpath_component_vocab=path_component_vocab,
            hidden_dim_lstm_xpath=xpath_lstm_dim,
            hidden_dim_lstm_word=word_lstm_dim,
            dropout=word_dropout,
            dropout_xpath=xpath_dropout,
        )
        super().__init__(
            tree_model=model,
            output_dimension=output_dimension,
            latent_dimension=latent_dim,
            input_dimension=input_dimension,
        )


@gin.configurable(allowlist=["local_module_hyper_parameters", "pretrained_word_embedding_model", "languages"])
class FreeDOMClassifier(TreeClassifier):
    """First stage of FreeDOM model for tree structured data"""

    def __init__(
        self,
        input_dimension,
        output_dimension,
        latent_dim,
        local_module_hyper_parameters: LocalModuleHyperParameters,
        pretrained_word_embedding_model: Union[Dict[str, Any], str],  # pylint: disable=unused-argument
        languages: List[str] = None,
        word_vocabulary=None,
        max_word_length=None,
        character_vocabulary=None,
    ):

        if word_vocabulary is None:
            pretrained_word_embeddings, _ = get_pretrained_fasttext_word_embeddings()
            word_vocabulary = list(pretrained_word_embeddings.keys())

        if max_word_length is None:
            max_word_length = get_max_word_length()

        if character_vocabulary is None:
            character_vocabulary = list(get_character_indexer().keys())

        if languages is None:
            languages = ["en"]

        # word_embedding_model: Dict[str, Any]
        # if isinstance(pretrained_word_embedding_model, dict):
        #     word_embedding_model = pretrained_word_embedding_model
        # else:
        #     if pretrained_word_embedding_model == "glove":
        #         word_embedding_model = load_glove_model(local_module_hyper_parameters.word_embedding_dim)
        #     elif pretrained_word_embedding_model == "fasttext":
        #         word_embedding_model = {}
        #         for language in languages:
        #             word_embedding_model.update(load_fasttext_model(language))
        #     elif pretrained_word_embedding_model == "treelstm-vocab":
        #         word_embedding_model = get_pretrained_word_embeddings()

        # TODO Decide on final structure for pre-trained word embeddings.

        model = FreeDOMModule(
            word_vocabulary=word_vocabulary,
            max_word_length=max_word_length,
            character_vocabulary=character_vocabulary,
            markup_features_count=len(TAGS_OF_INTEREST),
            string_type_features_count=len(FreeDOMFeatures.SPACY_FEATURES),
            pretrained_word_embedding_model=None,
            local_module_hyper_parameters=local_module_hyper_parameters,
            feedforward_layer_hidden_dim=latent_dim,
        )
        super().__init__(
            tree_model=model,
            output_dimension=output_dimension,
            latent_dimension=latent_dim,
            input_dimension=input_dimension,
        )


@gin.configurable(allowlist=["relational_module_hyper_parameters"])
class FreeDOMStageTwoClassifier(TreePairClassifier):
    """
    Second stage of FreeDOM model for classifying node pairs.
    """

    def __init__(
        self,
        input_dimension,
        output_dimension,
        latent_dim,
        relational_module_hyper_parameters: RelationalModuleHyperParameters,
    ):

        model_file = Path(relational_module_hyper_parameters.local_node_embedding_model_path)
        if model_file.is_file():
            pretrained_tree_classifier = torch.load(model_file)
        else:
            raise AttributeError("TreePairClassifier needs a pre-trained tree classifier to be initialized.")

        tree_pair_model = NodeDistanceEmbeddingModule(
            xpath_embedding_dim=relational_module_hyper_parameters.xpath_embedding_dim,
            position_size=relational_module_hyper_parameters.position_size,
            position_embedding_dim=relational_module_hyper_parameters.position_embedding_dim,
            local_tree_classifier=pretrained_tree_classifier,
            xpath_vocabulary_size=len(TAGS_OF_INTEREST),
            xpath_lstm_hidden_dim=relational_module_hyper_parameters.xpath_lstm_hidden_dim,
            dropout_rate=relational_module_hyper_parameters.dropout_rate,
            feedforward_layer_hidden_dim=latent_dim,
        )

        super().__init__(
            tree_pair_model=tree_pair_model,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            latent_dimension=latent_dim,
            hyper_parameter_m=relational_module_hyper_parameters.m_top_uncertain_field_nodes,
            hyper_parameter_n=relational_module_hyper_parameters.label_voting_threshold,
        )


class TransformerEncoderClassifier(TreeClassifier):
    """Transformer Encoder for tree structure data"""

    def __init__(self, input_dimension: int, output_dimension: int, latent_dimension: int):
        model = TransformerEncoderModule(input_dimension, latent_dimension=latent_dimension, n_head=5)
        super().__init__(model, model.padded_dimension, output_dimension, model.padded_dimension)

    def initialize(self):
        pass
