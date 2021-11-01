"""
Module containing the Text Embedding model and all of the layers.
"""
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional
from torch import nn


class PreTrainedWordEmbeddingLayer(nn.Module):
    """
    Class for pre-trained word embedding layer.
    """

    def __init__(self, output_dimension, word_vocabulary: List[str], model: Dict[str, Any]):
        super().__init__()

        self.output_dimensions = output_dimension
        self.pretrained_embedding = torch.nn.Embedding.from_pretrained(
            self.construct_weight_tensor(word_vocabulary, output_dimension, model)
        )

    def forward(self, word_sequence: torch.LongTensor):
        """
        Forward pass of model.

        :param word_sequence: 1D tensor containing word indices.
        :return: Output of forward pass.
        """
        return self.pretrained_embedding(word_sequence)

    @staticmethod
    def construct_weight_tensor(
        word_vocabulary: List[str], embedding_dimension: int, model: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Method that receives a word vocabulary and embedding dimension and returns a pre-trained weight matrix.
        Words that are in the vocabulary but not in the pre-trained model will be represented by a random vector.

        :param word_vocabulary: A list of words that are in the vocabulary.
        :param embedding_dimension: The embedding dimension of the word vectors.
        :param model: Pre-trained model to convert to weight tensor in dictionary format.
        :return: A weight matrix with shape (vocabulary length x embedding dimension)
        """
        unknown_embedding = torch.rand(embedding_dimension).tolist()
        vector_list = [model.get(word, unknown_embedding) for word in word_vocabulary]
        del model
        return torch.FloatTensor(vector_list)


class CharacterLevelWordEmbeddingLayer(nn.Module):
    """
    Class for the character embedding layer.
    """

    def __init__(
        self,
        character_embedding_dimension,
        character_vocabulary,
        max_word_length,
    ):
        super().__init__()
        self.max_word_length = max_word_length
        self.character_embedding_dimension = character_embedding_dimension
        self.character_embedding_layer = torch.nn.Embedding(
            num_embeddings=len(character_vocabulary),
            embedding_dim=self.character_embedding_dimension,
            padding_idx=0,
        )
        range_start = -math.sqrt(3 / self.character_embedding_dimension)
        range_end = math.sqrt(3 / self.character_embedding_dimension)
        nn.init.uniform_(self.character_embedding_layer.weight, range_start, range_end)
        self.character_embedding_layer.weight.data[0] = 0

    def forward(self, word_sequence: torch.LongTensor):
        """
        Forward pass of model.

        :param word_sequence: shape (words * max_word_length * character_embedding_dimension)
        """
        return self.character_embedding_layer(word_sequence)

    @staticmethod
    def pad_with_constant(word_character_index_list, max_length, pad_value) -> List[int]:
        """
        Center pad word indexes with padding index.

        :param word_character_index_list: List of characters equivalent to word on a character level)
        :param max_length: Maximum word length
        :param pad_value: The value to use for padding
        :return: Padded word on a character level
        """
        word_length = len(word_character_index_list)
        left_pad = math.ceil((max_length - word_length) / 2)
        right_pad = max_length - word_length - left_pad
        return [pad_value] * left_pad + word_character_index_list + [pad_value] * right_pad


class CNNWordEmbeddingLayer(nn.Module):
    """
    Class for the character-level word embedding layer.
    """

    def __init__(
        self,
        character_embedding_dimension,
        character_vocabulary,
        max_word_length,
        cnn_kernel_size=3,
        cnn_kernel_count=50,
    ):
        super().__init__()
        self.cnn_output_dimension = max_word_length - cnn_kernel_size + 1
        self.layer_output_dimension = character_embedding_dimension * cnn_kernel_count

        self.character_level_word_embedding_layer = CharacterLevelWordEmbeddingLayer(
            character_embedding_dimension=character_embedding_dimension,
            character_vocabulary=character_vocabulary,
            max_word_length=max_word_length,
        )

        self.cnn_1d_layer = torch.nn.Conv1d(
            in_channels=character_embedding_dimension,
            out_channels=character_embedding_dimension * cnn_kernel_count,
            kernel_size=cnn_kernel_size,
            groups=character_embedding_dimension,
        )
        self.max_pooling_layer = torch.nn.MaxPool1d(self.cnn_output_dimension)

    def forward(self, character_level_word_sequence: torch.LongTensor):
        """
        Forward pass of model.

        :param character_level_word_sequence: 2D Tensor containing indices of characters in words.
            shape: (word_count * max word length)
        """
        words_character_level = self.character_level_word_embedding_layer(character_level_word_sequence)
        words_character_level = words_character_level.permute(0, 2, 1)
        return torch.squeeze(self.max_pooling_layer(self.cnn_1d_layer(words_character_level)), dim=-1)


class TextEmbeddingModule(nn.Module):
    """
    Class containing the whole FreeDOM text embedding module.
    """

    def __init__(
        self,
        character_embedding_dim,
        word_embedding_dim,
        cnn_character_embedding_kernel_size,
        cnn_character_embedding_filter_count,
        rnn_hidden_dimension,
        character_vocabulary: List[str],
        word_vocabulary: List[str],  # pylint: disable=unused-argument
        max_word_length: int,
        pretrained_word_embedding_model: Dict[str, Any],  # pylint: disable=unused-argument
    ):
        super().__init__()

        # self.word_embedding_layer = PreTrainedWordEmbeddingLayer(
        #     output_dimension=word_embedding_dim,
        #     word_vocabulary=word_vocabulary, model=pretrained_word_embedding_model
        # )

        self.word_embedding_dim = word_embedding_dim
        self.character_embedding_dim = character_embedding_dim

        self.character_embedding_layer = CNNWordEmbeddingLayer(
            character_embedding_dimension=character_embedding_dim,
            character_vocabulary=character_vocabulary,
            max_word_length=max_word_length,
            cnn_kernel_size=cnn_character_embedding_kernel_size,
            cnn_kernel_count=cnn_character_embedding_filter_count,
        )

        self.bidirectional_lstm_layer = nn.LSTM(
            input_size=self.character_embedding_layer.layer_output_dimension + word_embedding_dim,
            hidden_size=rnn_hidden_dimension,
            batch_first=True,
            bidirectional=True,
        )

        self.output_dimension = 2 * rnn_hidden_dimension

    def forward(self, word_sequence, word_character_level_sequence):
        """
        Forward pass of model.

        :param word_sequence: shape (words * max_word_length * character_embedding_dimension)
        :param word_character_level_sequence: 2D Tensor containing indices of characters in words.
            shape: (word_count * max word length)
        """
        # word_embeddings = self.word_embedding_layer(word_sequence)
        word_embeddings = word_sequence
        character_level_word_embeddings = self.character_embedding_layer(word_character_level_sequence)
        word_encodings = torch.cat([word_embeddings, character_level_word_embeddings], dim=1)
        word_encodings = torch.unsqueeze(word_encodings, dim=0)
        lstm_output, _ = self.bidirectional_lstm_layer(word_encodings)
        node_text_embedding = torch.mean(lstm_output, dim=1)

        return node_text_embedding

    def forward_batch(self, word_sequence_batch, word_character_level_sequence_batch):
        """
        Forward pass of model for batched inputs

        :param word_sequence: shape (words * max_word_length * character_embedding_dimension)
        :param word_character_level_sequence: 2D Tensor containing indices of characters in words.
            shape: (word_count * max word length)
        """

        character_level_word_embeddings = [
            self.character_embedding_layer(seq) for seq in word_character_level_sequence_batch
        ]

        # word_embeddings_batch = [
        #     self.word_embedding_layer(word_sequence) for word_sequence in word_sequence_batch
        # ]
        word_embeddings_batch = word_sequence_batch

        word_encodings_batch = [
            torch.cat([word_embedding, char_word_embeddings], dim=1)
            for word_embedding, char_word_embeddings in zip(word_embeddings_batch, character_level_word_embeddings)
        ]

        packed_sequence = torch.nn.utils.rnn.pack_sequence(word_encodings_batch, enforce_sorted=False)

        lstm_output, _ = self.bidirectional_lstm_layer(packed_sequence)

        unpacked_output = self.unpack(lstm_output)

        node_text_embeddings = torch.stack([torch.mean(tens, dim=0) for tens in unpacked_output])

        return node_text_embeddings

    @staticmethod
    def unpack(packed_sequence):
        return [
            tnsr[:ln] for tnsr, ln in zip(*torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True))
        ]
