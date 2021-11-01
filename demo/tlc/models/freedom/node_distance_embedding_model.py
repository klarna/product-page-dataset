"""
Module containing freeDOM <https://arxiv.org/abs/2010.10755> second stage implementation.
"""
import torch
from torch import nn

from tlc.models.prototypes import TreeClassifier, TreeEmbedder
from tlc.structures.trees import NodePair


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


class XPathEncoderLayer(nn.Module):
    """
    Layer for embedding xpath sequences using a BiLSTM.
    """

    def __init__(self, xpath_vocabulary_size, xpath_embedding_dim, xpath_lstm_hidden_dim):
        super().__init__()

        self.output_dim = 2 * xpath_lstm_hidden_dim
        self.embeddings = nn.Embedding(xpath_vocabulary_size, xpath_embedding_dim)
        self.lstm_layer = nn.LSTM(xpath_embedding_dim, xpath_lstm_hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, xpath_sequence):
        """
        Forward pass of the model

        :param xpath_sequence: Sequence of xpaths. Should be a 1D Tensor with the indexes.
        :return: Output of forward pass.
        """
        embeddings = self.embeddings(xpath_sequence)
        embeddings = torch.unsqueeze(embeddings, dim=0)
        lstm_output, _ = self.lstm_layer(embeddings)
        xpath_embedding = torch.mean(lstm_output, dim=1)

        return xpath_embedding

    def forward_batch(self, xpath_sequence_batch):
        """
        Forward pass of model for batches of xpaths.

        :param xpath_sequence_batch:  Batcho of xpath sequences.
        :return: Output of forward pass.
        """

        embeddings_batch = [self.embeddings(xpath_sequence) for xpath_sequence in xpath_sequence_batch]

        packed_sequence = torch.nn.utils.rnn.pack_sequence(embeddings_batch, enforce_sorted=False)

        lstm_output, _ = self.lstm_layer(packed_sequence)

        unpacked_output = self.unpack(lstm_output)

        xpath_embeddings = torch.stack([torch.mean(tens, dim=0) for tens in unpacked_output])

        return xpath_embeddings

    @staticmethod
    def unpack(packed_sequence):
        return [
            tnsr[:ln] for tnsr, ln in zip(*torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True))
        ]


class NodePairRelationModule(nn.Module):
    """
    Layer for the embedding the distance of two nodes.
    """

    def __init__(
        self,
        xpath_embedding_dim,
        position_size,
        position_embedding_dim,
        xpath_vocabulary_size,
        xpath_lstm_hidden_dim,
    ):
        super().__init__()

        self.xpath_embedding_layer = XPathEncoderLayer(
            xpath_vocabulary_size=xpath_vocabulary_size,
            xpath_embedding_dim=xpath_embedding_dim,
            xpath_lstm_hidden_dim=xpath_lstm_hidden_dim,
        )
        self.position_embedding_layer = nn.Embedding(position_size, position_embedding_dim)

        self.output_dimension = 2 * (self.xpath_embedding_layer.output_dim + position_embedding_dim)

    def forward(self, head_node_xpath_seq, tail_node_xpath_seq, head_node_position, tail_node_position):
        """
        Forward pass of model.

        :param head_node_xpath_seq: Sequence of xpaths for the head node. Should be a 1D Tensor with the indexes.
        :param tail_node_xpath_seq: Sequence of xpaths for the head node. Should be a 1D Tensor with the indexes.
        :param head_node_position: Position of head node.
        :param tail_node_position:  Position of tail node.
        :return: Output of forward pass.
        """
        head_xpath_embedding = torch.squeeze(self.xpath_embedding_layer(head_node_xpath_seq))
        tail_xpath_embedding = torch.squeeze(self.xpath_embedding_layer(tail_node_xpath_seq))
        head_position_embedding = self.position_embedding_layer(head_node_position)
        tail_position_embedding = self.position_embedding_layer(tail_node_position)
        return torch.cat([head_xpath_embedding, tail_xpath_embedding, head_position_embedding, tail_position_embedding])

    def forward_batch(self, head_node_xpath_seq, tail_node_xpath_seq, head_node_position, tail_node_position):
        """
        Forward pass of model for batch.

        :param head_node_xpath_seq: Sequence of xpaths for the head node. Should be a 1D Tensor with the indexes.
        :param tail_node_xpath_seq: Sequence of xpaths for the head node. Should be a 1D Tensor with the indexes.
        :param head_node_position: Position of head node.
        :param tail_node_position:  Position of tail node.
        :return: Output of forward pass.
        """

        head_xpath_embedding_batch = self.xpath_embedding_layer.forward_batch(head_node_xpath_seq)
        tail_xpath_embedding_batch = self.xpath_embedding_layer.forward_batch(tail_node_xpath_seq)

        head_position_embedding_batch = self.position_embedding_layer(torch.stack(head_node_position))
        tail_position_embedding_batch = self.position_embedding_layer(torch.stack(tail_node_position))

        return torch.cat(
            [
                head_xpath_embedding_batch,
                tail_xpath_embedding_batch,
                head_position_embedding_batch,
                tail_position_embedding_batch,
            ],
            dim=1,
        )


class NodeDistanceEmbeddingModule(TreeEmbedder):
    """
    Module that contains the second stage of FreeDOM.
    """

    def __init__(
        self,
        xpath_embedding_dim,
        position_size,
        position_embedding_dim,
        local_tree_classifier: TreeClassifier,
        xpath_vocabulary_size,
        xpath_lstm_hidden_dim,
        dropout_rate,
        feedforward_layer_hidden_dim,
    ):
        super().__init__()

        self.node_pair_relation_layer = NodePairRelationModule(
            xpath_embedding_dim=xpath_embedding_dim,
            position_size=position_size,
            position_embedding_dim=position_embedding_dim,
            xpath_vocabulary_size=xpath_vocabulary_size,
            xpath_lstm_hidden_dim=xpath_lstm_hidden_dim,
        )

        self.local_tree_classifier = local_tree_classifier
        self.local_tree_classifier.requires_grad_(False)

        self.dropout_layer = torch.nn.Dropout(p=dropout_rate)

        self.pair_node_mlp = FeedForwardLayer(
            2 * self.local_tree_classifier.tree_model.node_embedding_dimension  # type: ignore
            + self.node_pair_relation_layer.output_dimension,
            feedforward_layer_hidden_dim,
        )

        self.output_dimension = feedforward_layer_hidden_dim

    def forward(self, node_pair: NodePair):  # pylint: disable=arguments-differ
        """
        Forward pass of module.

        :param node_pair: A pair of nodes.
        :return: Node pair embedding.
        """

        head_feature_tensors = node_pair.head_node.feature_vector.tensors
        tail_feature_tensors = node_pair.tail_node.feature_vector.tensors

        node_pair_relational_embedding = self.node_pair_relation_layer(
            head_node_xpath_seq=head_feature_tensors.feature_xpath,
            tail_node_xpath_seq=tail_feature_tensors.feature_xpath,
            head_node_position=head_feature_tensors.feature_location,
            tail_node_position=tail_feature_tensors.feature_location,
        )

        return self.dropout_layer(
            self.pair_node_mlp(
                torch.cat(
                    [
                        head_feature_tensors.feature_node_embedding,
                        tail_feature_tensors.feature_node_embedding,
                        node_pair_relational_embedding,
                    ]
                )
            )
        )

    def forward_batch(self, batch):  # pylint: disable=arguments-differ

        head_location_batch = []
        tail_location_batch = []

        head_xpath_batch = []
        tail_xpath_batch = []

        head_embedding_batch = []
        tail_embedding_batch = []

        for node_pair in batch:

            node_head_tensors = node_pair.head_node.feature_vector.tensors
            node_tail_tensors = node_pair.tail_node.feature_vector.tensors

            head_location_batch.append(node_head_tensors.feature_location)
            tail_location_batch.append(node_tail_tensors.feature_location)

            head_xpath_batch.append(node_head_tensors.feature_xpath)
            tail_xpath_batch.append(node_tail_tensors.feature_xpath)

            head_embedding = node_head_tensors.feature_node_embedding
            tail_embedding = node_tail_tensors.feature_node_embedding

            head_embedding_batch.append(head_embedding)
            tail_embedding_batch.append(tail_embedding)

        node_pair_relational_embedding_batch = self.node_pair_relation_layer.forward_batch(
            head_node_xpath_seq=head_xpath_batch,
            tail_node_xpath_seq=tail_xpath_batch,
            head_node_position=head_location_batch,
            tail_node_position=tail_location_batch,
        )

        return self.dropout_layer(
            self.pair_node_mlp(
                torch.cat(
                    [
                        torch.stack(head_embedding_batch),
                        torch.stack(tail_embedding_batch),
                        node_pair_relational_embedding_batch,
                    ],
                    dim=1,
                )
            )
        )

    def reset(self):
        pass
