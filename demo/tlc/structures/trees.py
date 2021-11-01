"""Classes representing tree-structured data"""

from __future__ import annotations

from collections import deque
from itertools import chain
from typing import Any, Iterator, List, NamedTuple, Optional, Tuple, TypeVar

import webtraversallibrary as wtl
from torch.types import Device

from ..models.freedom.data import FreeDOMFeatureMixin, FreeDOMStageTwoFeatureMixin
from ..models.riser.data_utils import RiSERFeatureMixin
from ..utilities import all_equal, local_cache
from .mixins import ElementMixin, FeatureMixin, LabelMixin, UniversalSentenceFeatureMixin

TreeType = TypeVar("TreeType", bound="Tree")


class Tree:
    """
    Base class for trees, contains all the logic for handling tree-structured data.
    """

    def __init__(self: TreeType, parent: Optional[TreeType] = None, children: Optional[List[TreeType]] = None):
        self.parent: Optional[TreeType] = parent
        self.children: List[TreeType] = [] if not children else children

    @property  # type: ignore
    @local_cache
    def root(self: TreeType) -> TreeType:
        """Returns the root of the tree"""
        return self if not self.parent else self.parent.root

    @property  # type: ignore
    @local_cache
    def depth(self: TreeType) -> int:
        """Returns the depth of the tree"""
        return 0 if not self.parent else 1 + self.parent.depth

    @property  # type: ignore
    @local_cache
    def n_leaves(self: TreeType) -> int:
        """Returns the number of leaves in the tree"""
        return 1 if not self.children else sum(child.n_leaves for child in self.children)

    @property  # type: ignore
    @local_cache
    def height(self: TreeType) -> int:
        """Returns  the height of the tree"""
        return 0 if not self.children else 1 + max(child.height for child in self.children)

    @property  # type: ignore
    @local_cache
    def non_structural_height(self) -> int:
        """Returns the height of the Tree disregarding single-child nodes"""
        if not self.children:
            return 0
        return (0 if len(self.children) <= 1 else 1) + max(child.non_structural_height for child in self.children)

    @property  # type: ignore
    @local_cache
    def size(self: TreeType) -> int:
        """Returns the number of nodes in the subtree rooted at this node."""
        return 1 + sum(child.size for child in self.children)

    @property
    def neighbors(self: TreeType) -> Iterator[TreeType]:
        """Returns all first-order neighbors of the node in the order
        [parent, children...]"""
        if self.parent is not None:
            yield self.parent
        yield from self.children

    def add_child(self: TreeType, child: TreeType) -> None:
        """Adds the child to the current subtree"""
        child.parent = self
        self.children.append(child)

    def __iter__(self: TreeType) -> Iterator[TreeType]:
        """Iterate over all nodes in the tree (in BFS fashion)"""
        q = [self]
        while q:
            node = q.pop(0)
            yield node
            q.extend(node.children)

    def distance(self: TreeType, node1: TreeType, node2: TreeType) -> int:
        """Returns the number of hops between node1 and node2"""

        path_to_nodes: Tuple[List[TreeType], List[TreeType]] = ([], [])

        for i, node in enumerate((node1, node2)):
            head = node
            while head is not self:
                path_to_nodes[i].append(head)
                head = head.parent
        path_to_node1, path_to_node2 = path_to_nodes

        dist = len(path_to_node1) + len(path_to_node2)
        while path_to_node1 and path_to_node2 and path_to_node1.pop() is path_to_node2.pop():
            dist -= 2

        return dist

    def single_child_path_lengths(self, path_length: int = 0) -> List[int]:
        """Returns a list of all single child path lengths in the Tree"""
        if len(self.children) == 1:
            return self.children[0].single_child_path_lengths(path_length + 1)
        return ([path_length] if path_length != 0 else []) + list(
            chain.from_iterable(child.single_child_path_lengths() for child in self.children)
        )

    def __str__(self: TreeType) -> str:
        return f"Tree({self.children})"

    def __eq__(self: TreeType, other: object) -> bool:
        if not isinstance(other, Tree):
            return NotImplemented

        return hash(self) == hash(other)

    def __hash__(self: TreeType) -> int:
        return id(self)

    def __repr__(self: TreeType) -> str:
        return str(self)

    def __contains__(self: TreeType, node: TreeType) -> bool:
        return self == node or any(node in child for child in self.children)


class DataTree(Tree, FeatureMixin, LabelMixin):
    """
    Class representing the DOM tree structure with a feature and a label.
    Feature and label transformations are passed in as mixin classes to allow for easy derived classes.
    """

    KLARNAI_ID_COLUMN = "wtl-uid"
    KLARNAI_PARENT_COLUMN = "wtl-parent-uid"

    def __init__(self, label: str = None, feature_vector: Any = None, element_index: int = None):
        super().__init__()
        self.label: str = label
        self.feature_vector: Any = feature_vector
        self.element_index: int = element_index  # Contains the original index of the element in element_metadata.

    @property
    def labeled(self) -> Iterator[DataTree]:
        return filter(lambda x: x.label not in [super().DEFAULT_LABEL, super().HARD_LABEL], iter(self))

    @property
    def unlabeled(self) -> Iterator[DataTree]:
        return filter(lambda x: x.label == super().DEFAULT_LABEL, iter(self))

    @property
    def unlabeled_hard(self) -> Iterator[DataTree]:
        return filter(lambda x: x.label == super().HARD_LABEL, iter(self))

    @classmethod
    def from_elements(cls, elements: wtl.Elements) -> Optional[DataTree]:
        """Construct a DataTree from a list of elements"""
        node_list: List[DataTree] = []

        root_to_node_paths: List[deque] = []  # Saves paths to labeled nodes

        for element_index, element in enumerate(elements):
            node = cls(super()._label(element), super()._features(element), element_index)
            parent = int(element.metadata["attributes"][cls.KLARNAI_PARENT_COLUMN])

            if parent != -1:
                node_list[parent].add_child(node)

            node_list.append(node)

            if node.label != super().DEFAULT_LABEL:
                root_to_node_paths.append(deque())

                head: Optional[Tree] = node
                while head:
                    root_to_node_paths[-1].append(head)
                    head = head.parent

        try:
            # Label the subject node by checking all root_to_node paths for overlap
            while all_equal(path[-1] for path in root_to_node_paths):
                subject_node = root_to_node_paths[-1][-1]
                for path in root_to_node_paths:
                    path.pop()

            subject_node.label = "subjectnode"  # The last overlapping node is the subjectnode

            return node_list[0]
        except IndexError:
            return None

    @classmethod
    def from_snapshot(cls, snapshot: wtl.PageSnapshot) -> Optional[DataTree]:
        """Construct a DataTree from a wtl snapshot"""
        return cls.from_elements(snapshot.elements)

    def __str__(self) -> str:
        return f"DataTree({self.label}, {self.children})"


class NodePair(NamedTuple):
    """
    Class representing a node pair and their pair label.
    """

    head_node: DataTree
    tail_node: DataTree
    head_label: str
    tail_label: str

    @property
    def label(self):
        return f"{self.head_label}-{self.tail_label}"


class TensorTree(DataTree):
    """Datatree where nodes have tensors as features"""

    def to(self, device: Device) -> None:
        """Iteratively move tree to the device"""
        for node in self:
            node.feature_vector.to(device)


class RiSERTree(DataTree, RiSERFeatureMixin):
    """Datatree for RiSER model"""


class FreeDOMDataTree(TensorTree, FreeDOMFeatureMixin):
    pass


class UniversalSentenceTree(DataTree, UniversalSentenceFeatureMixin):
    """
    Class representing the DOM tree structure with features based on the local text of a node.
    The local text is embedded by a pre-trained Universal Sentence Encoder.
    """
    pass


class FreeDOMStageTwoDataTree(TensorTree, FreeDOMStageTwoFeatureMixin):
    pass


class ElementTree(DataTree, ElementMixin):
    @property
    def element(self):
        return self.feature_vector
