from collections import Counter, defaultdict
from dataclasses import dataclass

from tlc.evaluate import compute_distance_to_true_nodes, evaluate_prediction, update_counts
from tlc.structures.trees import DataTree


def test_evaluate_prediction():
    @dataclass
    class _N:
        label: str

    predictions = {
        "a": [(0.99, _N("a")), (0.98, _N("b")), (0.97, _N("c"))],
        "b": [(0.99, _N("a")), (0.98, _N("b")), (0.97, _N("c"))],
        "c": [(0.99, _N("a")), (0.98, _N("b")), (0.97, _N("c"))],
    }

    n_correct, n_total = evaluate_prediction(predictions, [1, 2, 3])

    assert n_correct["a"] == {1: 1, 2: 1, 3: 1}
    assert n_correct["b"] == {1: 0, 2: 1, 3: 1}
    assert n_correct["c"] == {1: 0, 2: 0, 3: 1}

    assert n_total["a"] == {1: 1, 2: 1, 3: 1}
    assert n_total["b"] == {1: 1, 2: 1, 3: 1}
    assert n_total["c"] == {1: 1, 2: 1, 3: 1}


def test_update_counts():

    n_correct = defaultdict(Counter, {"a": Counter({1: 2, 2: 2, 7: 3}), "b": Counter({1: 0, 2: 1, 7: 4})})

    new_correct = defaultdict(Counter, {"a": Counter({1: 3, 7: 10}), "c": Counter({1: 2, 5: 6})})

    updated_correct = update_counts(n_correct, new_correct)

    assert updated_correct is n_correct
    assert n_correct["a"] == {1: 5, 2: 2, 7: 13}
    assert n_correct["b"] == {1: 0, 2: 1, 7: 4}
    assert n_correct["c"] == {1: 2, 5: 6}


def test_compute_distance_to_true_nodes():
    #        a
    #      / | \
    #     b  c  d
    #    / \  \
    #   e   f  g
    #      /
    #     h

    a = DataTree(DataTree.DEFAULT_LABEL)
    b = DataTree("b")
    c = DataTree(DataTree.DEFAULT_LABEL)
    d = DataTree("d")
    e = DataTree(DataTree.DEFAULT_LABEL)
    f = DataTree(DataTree.DEFAULT_LABEL)
    g = DataTree("g")
    h = DataTree("h")

    a.add_child(b)
    a.add_child(c)
    a.add_child(d)
    b.add_child(e)
    b.add_child(f)
    f.add_child(h)
    c.add_child(g)

    assert sorted(list(x.label for x in a.labeled)) == ["b", "d", "g", "h"]

    predictions = {
        "b": [(0.99, b)],
        "d": [(0.99, g)],
        "g": [(0.99, h)],
        "h": [(0.99, a)],
    }

    distances = compute_distance_to_true_nodes(predictions, a)

    assert distances["b"] == 0
    assert distances["d"] == 3
    assert distances["g"] == 5
    assert distances["h"] == 3
