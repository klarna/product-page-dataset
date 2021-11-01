"""Base classes for element classifiers"""

import heapq
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Collection, Dict, List, Tuple, Type, Union

import numpy
import torch
from torch import Tensor, nn
from tqdm import tqdm

from ..structures.mixins import LabelMixin, PairLabelMixin
from ..structures.trees import DataTree, NodePair
from .utilities import SaveMixin

TreePredictType = Dict[Union[str, int], List[Tuple[float, DataTree]]]
NodePredictionType = Tuple[float, int, DataTree]


class TreeEmbedder(nn.Module, ABC):
    """Base class for tree embedders"""

    def __init__(self, **dimensions: int):  # pylint: disable=unused-argument
        super().__init__()

    @abstractmethod
    def forward(self, node) -> torch.Tensor:
        """Define the forward pass of the embedder
        Takes a tree-node and returns a tensor embedding"""

    def forward_batch(self, node_batch) -> torch.Tensor:
        """Define the batch forward pass of the embedder
        Takes a batch of tree-node's and returns a tensor embedding"""
        return torch.stack([self.forward(node) for node in node_batch])

    def reset(self) -> None:
        """Resets eventual model caches"""


class TreeClassifier(SaveMixin, nn.Module, LabelMixin):
    """Base class for classifiers for tree-structured data"""

    def __init__(
        self,
        tree_model: TreeEmbedder,
        input_dimension: int,
        output_dimension: int,
        latent_dimension: int,
    ):
        super().__init__()
        self.tree_model = tree_model
        self.output = nn.Linear(latent_dimension, output_dimension)
        self.softmax = nn.Softmax(dim=0)
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.output_dimension = output_dimension

    def forward(self, node: DataTree) -> Tensor:
        """Forward pass of the model"""
        h = self.tree_model(node)
        return self.output(h)

    def forward_batch(self, batch):
        h = self.tree_model.forward_batch(batch)
        return self.output(h)

    def predict_proba(self, node: DataTree) -> List[float]:
        """Returns the posterior probabilities of the labels"""
        return self.softmax(self(node)).tolist()

    def predict_proba_batch(self, node: List[DataTree]) -> List[List[float]]:
        """Returns the posterior probabilities of the labels for batches of nodes"""
        return self.softmax(self.forward_batch(node)).tolist()

    def predict(self, node: DataTree, human_readable: bool = True) -> Union[str, int]:
        """Returns a prediction for the node in input; if human_readable,
        returns the string representation of the class."""
        class_probabilities = self.softmax(self(node))

        if human_readable:
            return self.label_decode(int(torch.argmax(class_probabilities).item()))

        return int(torch.argmax(class_probabilities).item())

    def predict_tree(
        self, tree: DataTree, n_candidates: int = 5, human_readable: bool = True, use_augmented=False
    ) -> TreePredictType:
        """Returns a dict where `dict[label]` is list of nodes that match `label` together with class probability"""

        labeled_nodes: Dict[Union[str, int], List[Tuple[float, int, DataTree]]] = defaultdict(list)
        node_batch = list(tree)
        if use_augmented:
            class_probs_batch = [node.feature_vector.feature_class_probabilities for node in node_batch]
        else:
            class_probs_batch = self.predict_proba_batch(node_batch)

        for idx, (node, class_probs) in enumerate(zip(node_batch, class_probs_batch)):
            for class_, prob in enumerate(class_probs):
                count = len(labeled_nodes[class_])
                _action: Callable = heapq.heappush if count < n_candidates else heapq.heappushpop
                _action(labeled_nodes[class_], (prob, idx, node))

        output_nodes: TreePredictType = dict()
        for prediction, candidates in labeled_nodes.items():
            best_predictors: List[Tuple[float, int, DataTree]] = heapq.nlargest(n_candidates, candidates)
            output_nodes[prediction] = [(prob, node) for prob, _, node in best_predictors]

        if human_readable:
            return {self.label_decode(int(k)): v for k, v in output_nodes.items()}

        return output_nodes

    def initialize(self) -> None:
        """Initialize the parameters of the model"""
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def reset(self) -> None:
        """Reset eventual model caches"""
        self.tree_model.reset()


def create_pair_indices(alist: List) -> List[Tuple[int, int]]:
    """
    Create list of indices to create pairs from a list excluding the pairs containing the same element.

    :param alist: List to create pair indices for.
    :return: List of pair indices.
    """
    pairs_indices: List[Tuple[int, int]] = []
    indices_range = range(len(alist))
    for head_index in indices_range:
        tail_index_start = head_index + 1
        for tail_index in indices_range[tail_index_start:]:
            pairs_indices.append((head_index, tail_index))
            pairs_indices.append((tail_index, head_index))
    return pairs_indices


class TreePairClassifier(SaveMixin, nn.Module, PairLabelMixin):
    """
    Base class for node pair classifiers for tree-structured data
    """

    def __init__(
        self,
        tree_pair_model,
        input_dimension: int,
        output_dimension,
        latent_dimension,
        hyper_parameter_m,
        hyper_parameter_n,
    ):
        super().__init__()

        self.tree_pair_model = tree_pair_model
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.output_dimension = output_dimension
        self.output = nn.Linear(latent_dimension, output_dimension)
        self.softmax = nn.Softmax(dim=0)
        self.hyper_parameter_m = hyper_parameter_m
        self.hyper_parameter_n = hyper_parameter_n

    @property
    def local_tree_classifier(self):
        return self.tree_pair_model.local_tree_classifier

    def collate_fn(self) -> Callable[[Collection[DataTree]], List[NodePair]]:
        """
        Returns a collate_fn that assembles batches of node pairs from batches of DataTrees


        :return: collate_fn that assembles batches of node pairs from batches of DataTrees
        """

        def _collate_fn(batch: Collection[DataTree]) -> List[NodePair]:

            training_node_pairs = []
            for tree in batch:
                if tree is None:
                    continue
                with torch.no_grad():
                    candidate_nodes, pair_indices, _, _, _ = self.get_candidate_pairs(tree)

                for head_index, tail_index in pair_indices:
                    _, (_, _, head_node) = candidate_nodes[head_index]
                    _, (_, _, tail_node) = candidate_nodes[tail_index]
                    training_node_pairs.append(self._create_pair(head_node, tail_node))

            random.shuffle(training_node_pairs)
            return training_node_pairs

        return _collate_fn

    def get_certain_uncertain_fields(
        self,
        tree: DataTree,
    ) -> Tuple[Dict[str, List[NodePredictionType]], Dict[str, List[NodePredictionType]]]:
        """
        Method that takes a tree as input and returns the uncertain and certain fields
        in the tree as output with their respective nodes.
        :param tree: The tree to gather certain and uncertain fields for.
        :return:
        Return two dictionaries for certain and uncertain fields.
        The values of the dictionaries are the respective nodes.
        """

        certain_fields: Dict[str, List[NodePredictionType]] = defaultdict(list)
        uncertain_fields: Dict[str, List[NodePredictionType]] = defaultdict(list)

        for idx, node in tqdm(enumerate(tree), leave=False, position=2, desc="Predicting Tree", total=tree.size):
            class_probabilities = node.feature_vector.feature_class_probabilities
            predicted_node_label_index = numpy.argmax(class_probabilities)
            predicted_node_label = self.local_tree_classifier.label_decode(predicted_node_label_index)

            if predicted_node_label != self.local_tree_classifier.DEFAULT_LABEL:
                # Add certain field and keep them sorted by probability.
                prob = class_probabilities[predicted_node_label_index]
                heapq.heappush(certain_fields[predicted_node_label], (prob, idx, node))
            else:
                # Keep track of the probability of the uncertain nodes.
                for class_, prob in enumerate(class_probabilities):
                    count = len(uncertain_fields[self.local_tree_classifier.label_decode(class_)])
                    _action: Callable = heapq.heappush if count < self.hyper_parameter_m else heapq.heappushpop
                    _action(uncertain_fields[self.local_tree_classifier.label_decode(class_)], (prob, idx, node))

        # Remove certain fields from the uncertain dictionary.
        for certain_field in certain_fields:
            uncertain_fields.pop(certain_field, None)

        # Remove default label from the uncertain dictionary.
        uncertain_fields.pop(self.local_tree_classifier.DEFAULT_LABEL, None)

        return certain_fields, uncertain_fields

    def get_candidate_pairs(
        self,
        tree: DataTree,
    ) -> Tuple[
        List[Tuple[str, NodePredictionType]],
        List[Tuple[int, int]],
        List[bool],
        Dict[str, List[NodePredictionType]],
        Dict[str, List[NodePredictionType]],
    ]:
        """
        Method that takes a tree as input and returns the candidate pairs alongside certain and uncertain fields.
        :param tree: The tree to gather certain and uncertain fields for.
        :return:
        A tuple with five members as follows.
        list of (label, node) elements which are the candidate nodes to create pairs from.
        list of (index1, index2) containing the indices of the pairs based on the candidate nodes list.
        uncertain node mask to be able to mask certain and uncertain nodes.
        dictionary of label to nodes for certain fields.
        dictionary of label to nodes for uncertain fields.
        """

        certain_fields, uncertain_fields = self.get_certain_uncertain_fields(tree)

        pair_candidate_nodes: Dict[Union[str], List[Tuple[float, int, DataTree]]] = defaultdict(list)
        uncertain_mask: List[bool] = []

        # Add the best node for each certain field to pair candidates.
        for certain_field, certain_nodes in certain_fields.items():
            pair_candidate_nodes[certain_field] = heapq.nlargest(1, certain_nodes)
            uncertain_mask += [False] * len(pair_candidate_nodes[certain_field])

        # Add top m nodes for the uncertain fields to pair candidates.
        for uncertain_field, uncertain_nodes in uncertain_fields.items():
            pair_candidate_nodes[uncertain_field] = heapq.nlargest(self.hyper_parameter_m, uncertain_nodes)
            uncertain_mask += [True] * len(pair_candidate_nodes[uncertain_field])

        # Flatten candidate pairs in one list.
        label_node_list: List[Tuple[str, NodePredictionType]] = []
        for predicted_label, nodes in pair_candidate_nodes.items():
            label_node_list += [(predicted_label, node) for node in nodes]

        # Create pair indices.
        pairs_indices: List[Tuple[int, int]] = create_pair_indices(label_node_list)

        return label_node_list, pairs_indices, uncertain_mask, certain_fields, uncertain_fields

    def _count_votes(
        self, candidate_nodes: List[Tuple[str, NodePredictionType]], pair_indices: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Method that takes a list of candidate nodes and pair indices as input and counts label, none votes for them.

        :param candidate_nodes: A list of candidate nodes.
        :param pair_indices: A list of index pairs determining the pairs to use for voting.
        :return: A list of votes where vote[i] is the number of LABEL votes for candidate_nodes[i]
        """
        node_votes: List[int] = [0] * len(candidate_nodes)

        for head_index, tail_index in tqdm(pair_indices, position=3, leave=False, desc="Predicting Tree Pairs"):
            _, (_, _, head_node) = candidate_nodes[head_index]
            _, (_, _, tail_node) = candidate_nodes[tail_index]

            head_pair_label, tail_pair_label = self.predict(
                self._create_pair(head_node, tail_node), human_readable=True, splitted=True
            )
            node_votes[head_index] += 1 if head_pair_label == self.VALUE_LABEL else 0
            node_votes[tail_index] += 1 if tail_pair_label == self.VALUE_LABEL else 0
        return node_votes

    def _create_pair(self, head_node: DataTree, tail_node: DataTree):
        """
        Create node pair from data tree nodes.

        :param head_node: Head node.
        :param tail_node: Tail node
        :return: Node pair object representing the node pairs and their pair labels.
        """
        return NodePair(
            head_node,
            tail_node,
            self.VALUE_LABEL if head_node.label in self.local_tree_classifier.NODE_LABELS else self.NONE_LABEL,
            self.VALUE_LABEL if tail_node.label in self.local_tree_classifier.NODE_LABELS else self.NONE_LABEL,
        )

    def predict_tree(
        self,
        tree: DataTree,
        n_candidates=5,
        human_readable=True,
        use_augmented=False,  # pylint: disable=unused-argument
    ) -> TreePredictType:
        """
        Method that takes a data tree as input and returns the class probabilities for the nodes.

        :param tree: The Data tree
        :param n_candidates: Number of candidates for each label to return
        :param human_readable: If set to True will return labels in string form.
        :param use_augmented: Set to true to use augmented predictions. Only works for TreeClassifier.
        :return: Returns a dict where `dict[label]` is list of nodes that match `label` together with class probability.
        """
        with torch.no_grad():
            candidate_nodes, pair_indices, uncertain_mask, certain_fields, uncertain_fields = self.get_candidate_pairs(
                tree
            )
        node_votes = self._count_votes(candidate_nodes, pair_indices)

        # Add the certain nodes as is to the final labelled nodes.
        labeled_nodes: Dict[Union[str, int], List[Tuple[float, int, DataTree]]] = defaultdict(list)
        for certain_field, certain_nodes in certain_fields.items():
            labeled_nodes[certain_field] = certain_nodes

        # Add uncertain nodes that passed the vote threshold to the final labelled nodes.
        for candidate_node, uncertain, votes in zip(candidate_nodes, uncertain_mask, node_votes):
            if not uncertain:
                continue
            tentative_label, (prob, tree_index, node) = candidate_node
            if votes >= self.hyper_parameter_n:
                final_label = tentative_label
                count = len(labeled_nodes[final_label])
                _action: Callable = heapq.heappush if count < n_candidates else heapq.heappushpop
                _action(labeled_nodes[final_label], (prob + votes, tree_index, node))
            else:
                final_label = self.local_tree_classifier.DEFAULT_LABEL
                count = len(labeled_nodes[final_label])
                _action = heapq.heappush if count < n_candidates else heapq.heappushpop
                _action(labeled_nodes[final_label], (prob, tree_index, node))

        # Take top n for each label as outputs of the model.
        output_nodes: Dict[Union[str, int], List[Tuple[float, DataTree]]] = dict()
        for prediction, candidates in labeled_nodes.items():
            best_predictors: List[Tuple[float, int, DataTree]] = heapq.nlargest(n_candidates, candidates)
            output_nodes[prediction] = [(prob, node) for prob, _, node in best_predictors]

        # Some uncertain fields might not pass the voting threshold and therefore be empty. For these fields we simply
        # return the top n nodes for that class sorted by their probability.
        for label in self.local_tree_classifier.ALL_LABELS:
            if label not in output_nodes:
                output_nodes[label] = [
                    (prob, node) for prob, _, node in heapq.nlargest(n_candidates, uncertain_fields[label])
                ]

        if not human_readable:
            return {self.label_encode(str(k)): v for k, v in output_nodes.items()}

        return output_nodes

    def predict(self, node_pair: NodePair, human_readable=True, splitted=False):
        """
        Returns a prediction for the node pair input; if human_readable,
        returns the string representation of the class.

        :param node_pair: A tuple of nodes
        :param human_readable: If set to true it will return string representation of label.
        :param splitted: If set to true the function returns the labels of the thread separated.
        :return:
        """

        class_probabilities = self.softmax(self(node_pair))

        label_index = int(torch.argmax(class_probabilities).item())

        if human_readable:
            label = self.label_decode(label_index)
            if splitted:
                return self.split_label(label)
            return label

        return int(label_index)

    def forward(self, node_pair: NodePair):
        h = self.tree_pair_model(node_pair)
        return self.output(h)

    def forward_batch(self, batch):
        h = self.tree_pair_model.forward_batch(batch)
        return self.output(h)

    def initialize(self):
        """Initialize the parameters of the model"""
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def reset(self) -> None:
        """Reset eventual model caches"""
        self.tree_pair_model.reset()


class BidirectionalModel(TreeEmbedder):
    """Base class for bidirectional models for tree-structured data"""

    def __init__(
        self, tree_models: Tuple[Type[TreeEmbedder], Type[TreeEmbedder]], input_dimension: int, output_dimension: int
    ):
        super().__init__()
        bottom_up_model, top_down_model = tree_models
        self.bottom_up_model = bottom_up_model(input_dimension, output_dimension)  # type: ignore
        self.top_down_model = top_down_model(input_dimension, output_dimension)  # type:ignore

    def forward(self, node: DataTree) -> Tensor:
        """Forward pass of the model"""
        h_bu = self.bottom_up_model(node)
        h_td = self.top_down_model(node)

        return torch.cat((h_bu, h_td))

    def reset(self) -> None:
        """Reset eventual model caches"""
        self.bottom_up_model.reset()
        self.top_down_model.reset()


class BidirectionalModelWithEmbeddings(TreeEmbedder):
    """Base class for bidirectional models for tree-structured data; in the second pass, the output of the first
    pass (the embedding) is used instead of the local features"""

    def __init__(
        self, tree_models: Tuple[Type[TreeEmbedder], Type[TreeEmbedder]], input_dimension: int, output_dimension: int
    ):
        super().__init__()
        bottom_up_model, top_down_model = tree_models
        self.bottom_up_model = bottom_up_model(input_dimension, output_dimension)  # type: ignore
        self.top_down_model = top_down_model(output_dimension, output_dimension)  # type: ignore

    def forward(self, node: DataTree) -> Tensor:
        """Forward pass of the model"""
        h_bu = self.bottom_up_model(node)
        h_td = self.top_down_model(node, tree_embedding=self.bottom_up_model)

        return torch.cat((h_bu, h_td))

    def reset(self) -> None:
        """Reset eventual model caches"""
        self.bottom_up_model.reset()
        self.top_down_model.reset()
