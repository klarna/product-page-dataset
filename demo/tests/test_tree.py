# pylint: disable=redefined-outer-name, comparison-with-itself
"""Tests for the Tree class"""

import pytest

from tlc.structures.trees import Tree


@pytest.fixture(scope="function")
def sample_tree():
    #        a
    #      / | \
    #     b  c  d
    #    / \  \
    #   e   f  g
    #      /
    #     h

    a = Tree()
    b = Tree()
    c = Tree()
    d = Tree()
    e = Tree()
    f = Tree()
    g = Tree()
    h = Tree()

    a.add_child(b)
    a.add_child(c)
    a.add_child(d)
    b.add_child(e)
    b.add_child(f)
    f.add_child(h)
    c.add_child(g)

    return a, b, c, d, e, f, g, h


def test_add_child():
    a = Tree()
    b = Tree()

    a.add_child(b)

    assert len(a.children) == 1
    assert a.children[0].parent == a
    assert a.children[0] == b


def test_height(sample_tree):
    a, _, c, _, _, _, g, _ = sample_tree

    assert a.height == 3
    assert c.height == 1
    assert g.height == 0


def test_depth(sample_tree):
    a, _, c, _, _, _, _, h = sample_tree

    assert a.depth == 0
    assert c.depth == 1
    assert h.depth == 3


def test_n_leaves(sample_tree):
    a, _, c, _, _, _, _, h = sample_tree
    assert a.n_leaves == 4
    assert c.n_leaves == 1
    assert h.n_leaves == 1


def test_non_structural_height(sample_tree):
    a, b, _, _, _, f, _, h = sample_tree

    assert a.non_structural_height == 2
    assert b.non_structural_height == 1
    assert f.non_structural_height == 0
    assert h.non_structural_height == 0


def test_single_child_path_lengths(sample_tree):
    a, b, _, _, _, f, _, h = sample_tree

    assert a.single_child_path_lengths() == [1, 1]
    assert b.single_child_path_lengths() == [1]
    assert f.single_child_path_lengths() == [1]
    assert h.single_child_path_lengths() == []


def test_size(sample_tree):
    a, b, _, _, _, _, g, _ = sample_tree

    assert a.size == 8
    assert b.size == 4
    assert g.size == 1


def test_root(sample_tree):
    a, _, _, _, _, f, g, _ = sample_tree
    assert f.root == g.root == a


def test_neighbors_singleton():
    a = Tree()

    assert len(list(a.neighbors)) == 0


def test_neighbors(sample_tree):
    a, b, c, d, e, f, g, h = sample_tree

    assert list(a.neighbors) == [b, c, d]
    assert list(b.neighbors) == [a, e, f]
    assert list(g.neighbors) == [c]
    assert list(h.neighbors) == [f]


def test_distance(sample_tree):
    a, b, c, d, e, f, g, h = sample_tree

    assert a.distance(b, d) == 2
    assert a.distance(a, h) == 3
    assert a.distance(e, h) == 3
    assert a.distance(e, g) == 4
    assert a.distance(c, g) == 1
    assert a.distance(c, f) == 3


def test_contains(sample_tree):
    a, b, c, _, _, _, g, h = sample_tree

    assert a in a
    assert b in a
    assert g in a
    assert b in b
    assert h in b
    assert a not in b
    assert h not in c


def test_iterable(sample_tree):
    a, b, c, d, e, f, g, h = sample_tree

    assert list(iter(a)) == [a, b, c, d, e, f, g, h]
