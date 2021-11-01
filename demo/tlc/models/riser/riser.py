"""
RiSER architecture as interpreted from:
https://ai.google/research/pubs/pub47858
(RiSER: Learning Better Representations for Richly Structured Emails)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from ..prototypes import TreeEmbedder
from .data import Vocabulary

if TYPE_CHECKING:
    from tlc.structures.trees import DataTree


class Attention(nn.Module):
    """
    Attention unit. Returns similarity scores between inputs and trainable lookup parameter.
    """

    def __init__(self, hidden_dim_attn: int):
        super().__init__()
        self.query = Parameter(torch.empty(hidden_dim_attn).uniform_())

    def forward(self, u_vecs: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """
        Calculate normed similarity scores between u vectors and
        structure vector/query
        :param u_vecs: Per batch (row), a batch_length x hidden_length vec
        :return: Per batch (row), a 1 x batch_length vec
        """
        scores = torch.matmul(self.query, u_vecs.transpose(-2, -1))
        scores = nn.functional.normalize(scores, dim=-1)  # Lp normalization, not the same as before
        return scores


def weighted_sum(weights: Tensor, vecs: Tensor) -> Tensor:
    """
    Batched weighted sum
    :param weights: Per batch (row), a 1 x batch_length vec
    :param vecs: Per batch (row), a batch_length x repr_length vec
    :return: Per batch (row), a 1 x repr_length vec
    """
    return torch.einsum("ijk,ij->ik", [vecs, weights])


class XPathEncoder(nn.Module):
    """
    Encoder for XPath sequences
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim_lstm: int, dropout: float):
        super().__init__()
        # OBS! hidden_dim_lstm is the embedding dimension for the entire xpath sequence!
        # embedding_dim is the dimension for each x path component
        self.dim = hidden_dim_lstm
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim_lstm, dropout=dropout)

        self.compute_u = nn.Sequential(nn.Linear(hidden_dim_lstm, hidden_dim_lstm), nn.Tanh())

        self.attention = Attention(hidden_dim_lstm)

    def forward(self, seq: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """Forward pass of the model"""
        embeds = self.embeddings(seq)

        # Get LSTM output vectors h for each t
        h, _ = self.lstm(embeds)  # Expected dimension: batch size, batch length, hidden lstm dim

        # Calculate hidden vector u for each t: u = tanh(w * h + b)
        u = self.compute_u(h)

        # Compute normalized importance weight a for each t
        a = self.attention(u)

        # Compute XPath encoding vector x as weighted sum of all h
        x = weighted_sum(a, h)

        return x


class WordEncoder(nn.Module):
    """
    To form improved word representations, the word embeddings,
    XPath encodings, and annotation vectors (later) are concatenated for each
    term and fed into a fully connected layer with tanh non-linearity.
    """

    def __init__(self, xpath_embed: XPathEncoder, word_embed: nn.Embedding):
        super().__init__()
        self.xpath_embed = xpath_embed
        self.word_embed = word_embed
        self.dim = self.word_embed.embedding_dim + self.xpath_embed.dim
        self.ff = nn.Linear(self.dim, self.dim)

    def forward(self, words: Tensor, xpaths: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """Forward pass of the model"""
        word_embeds = self.word_embed(words)  # Dim of word_embeds: batch_size x doc_length x word_dim

        # Find unique xpath sequences
        [batch_size, doc_len, xpath_len] = xpaths.size()  # Dim of xpaths: batch size x doc len x xpath len
        all_xpaths = xpaths.view(batch_size * doc_len, xpath_len)
        uniq_xpaths, idx_to_uniq = torch.unique(all_xpaths, return_inverse=True, dim=0)

        # Compute encodings for unique xpath sequences
        uniq_xpath_embeds = self.xpath_embed(uniq_xpaths)
        xpath_embeds = torch.index_select(uniq_xpath_embeds, dim=0, index=idx_to_uniq)
        xpath_embeds = xpath_embeds.view(batch_size, doc_len, self.xpath_embed.dim)  # Reshape

        # Interleave word and xpath embeddings
        full_embeds = torch.cat((word_embeds, xpath_embeds), dim=-1)
        return full_embeds


class RiSER(TreeEmbedder):
    """
    Final RiSER encoding.
    """

    def __init__(
        self,
        word_embed_dim: int,
        xpath_embed_dim: int,  # Embed hyperparameters
        word_vocab: Vocabulary,
        xpath_component_vocab: Vocabulary,
        hidden_dim_lstm_word: int = 128,  # LSTM hidden state dim hyperparams
        hidden_dim_lstm_xpath: int = 64,
        dropout: float = 0.25,
        dropout_xpath: float = 0.5,
    ):
        super().__init__()
        xpath_component_vocab_size = len(xpath_component_vocab)
        word_vocab_size = len(word_vocab)

        xpath_embed = XPathEncoder(xpath_component_vocab_size, xpath_embed_dim, hidden_dim_lstm_xpath, dropout_xpath)
        word_embed = nn.Embedding(word_vocab_size, word_embed_dim)

        self.embeddings = WordEncoder(xpath_embed, word_embed)
        embedding_dim = self.embeddings.dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim_lstm_word, dropout=dropout)

        self.hidden = hidden_dim_lstm_word
        self.compute_u = nn.Sequential(nn.Linear(hidden_dim_lstm_word, hidden_dim_lstm_word), nn.Tanh())

        self.attention = Attention(hidden_dim_lstm_word)

    def forward(self, node: DataTree) -> Tensor:
        """Forward pass of the model"""
        # Encode document
        words, xpaths = node.feature_vector
        embeddings = self.embeddings(words.unsqueeze(0), xpaths.unsqueeze(0))
        h, _ = self.lstm(embeddings)
        u = self.compute_u(h)
        a = self.attention(u)
        x = weighted_sum(a, h)
        return x.squeeze(0)
