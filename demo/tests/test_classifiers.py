# pylint: disable=redefined-outer-name
from __future__ import annotations

from itertools import zip_longest

import pytest
import torch
from torch.utils.data import DataLoader

from tlc.models.elementclassifiers import (
    BidirectionalLSTMClassifier,
    BidirectionalLSTMClassifierWithEmbeddings,
    BidirectionalRNNClassifier,
    BidirectionalRNNClassifierWithEmbeddings,
    BottomUpLSTMClassifier,
    DOMQNETClassifier,
    DOMQNETWithGlobalEmbeddingClassifier,
    FeedforwardMultiLayerGCNClassifier,
    FullyConnectedClassifier,
    GATClassifier,
    GATDotProductClassifier,
    GCNClassifier,
    MultiLayerGCNClassifier,
    TopDownLSTMClassifier,
    TransformerEncoderClassifier,
)
from tlc.structures.trees import DataTree
from tlc.trainers import SimpleTrainer

ALL_CLASSIFIERS = [
    BidirectionalLSTMClassifier,
    BidirectionalLSTMClassifierWithEmbeddings,
    BidirectionalRNNClassifier,
    BidirectionalRNNClassifierWithEmbeddings,
    BottomUpLSTMClassifier,
    DOMQNETClassifier,
    DOMQNETWithGlobalEmbeddingClassifier,
    FullyConnectedClassifier,
    GATClassifier,
    GATDotProductClassifier,
    GCNClassifier,
    MultiLayerGCNClassifier,
    FeedforwardMultiLayerGCNClassifier,
    TopDownLSTMClassifier,
    TransformerEncoderClassifier,
]


@pytest.mark.parametrize("classifier", ALL_CLASSIFIERS)
def test_save_load(classifier, tmp_path):

    path = tmp_path / "testmodel.pt"

    mdl = classifier(4, 3, latent_dimension=14)

    mdl.save(loc=path)

    loaded_mdl = classifier.load(loc=path)

    assert loaded_mdl is not mdl
    assert isinstance(loaded_mdl, classifier)
    for par1, par2 in zip_longest(mdl.parameters(), loaded_mdl.parameters()):
        assert torch.equal(par1, par2)  # pylint: disable=no-member


@pytest.mark.parametrize("classifier", ALL_CLASSIFIERS)
def test_classifier_overfit(classifier):

    torch.manual_seed(1337)

    a = DataTree("a", torch.tensor([1, 0, 0, 0]))  # pylint: disable=not-callable
    b = DataTree("b", torch.tensor([0, 1, 0, 0]))  # pylint: disable=not-callable
    c = DataTree("c", torch.tensor([0, 0, 1, 0]))  # pylint: disable=not-callable
    d = DataTree("c", torch.tensor([0, 0, 0, 1]))  # pylint: disable=not-callable

    a.add_child(b)
    a.add_child(c)
    b.add_child(d)

    class SimpleClassifier(classifier):
        ALL_LABELS = list("abc")

    model = SimpleClassifier(4, 3, latent_dimension=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = SimpleTrainer(model, optimizer, loss_fn)

    trainer.train(DataLoader([a, b, c, d], batch_size=4, collate_fn=list, num_workers=0), 100)

    assert model.predict(a) == "a"
    assert model.predict(b) == "b"
    assert model.predict(c) == "c"
    assert model.predict(d) == "c"
