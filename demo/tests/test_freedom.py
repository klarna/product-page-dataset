# pylint: disable=redefined-outer-name,protected-access
from dataclasses import dataclass
from itertools import zip_longest
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tlc.dataset.utilities import TAG_TO_INT
from tlc.models.elementclassifiers import FreeDOMClassifier, FreeDOMStageTwoClassifier
from tlc.models.freedom.data import FreeDOMFeatures, FreeDOMFeatureTensors
from tlc.models.freedom.model import LocalModuleHyperParameters, RelationalModuleHyperParameters
from tlc.structures.trees import FreeDOMDataTree
from tlc.trainers import SimpleTrainer


class SimpleStageOneClassifier(FreeDOMClassifier):
    NODE_LABELS = list("abc")
    ALL_LABELS = [FreeDOMClassifier.DEFAULT_LABEL] + NODE_LABELS


class SimpleStageTwoClassifier(FreeDOMStageTwoClassifier):
    pass


@pytest.fixture(scope="session")
def model_temp_path(tmpdir_factory):
    fn = tmpdir_factory.mktemp("model")
    return fn


# pylint: disable=not-callable
@dataclass
class FreeDOMFeatureVector:
    feature_text_indexes: List  # shape: word count
    feature_parent_text_indexes: List  # shape: word count
    feature_character_indexes: List  # shape: word count * max_word_length
    feature_parent_character_indexes: List  # shape: word count * max_word_length
    tag: List
    node_string_types: List
    feature_location: int
    feature_xpath: List
    feature_class_probabilities: List
    feature_node_embedding: List
    feature_tentative_label: str
    feature_common_features: List

    @property
    def tensors(self):
        return FreeDOMFeatureTensors(
            torch.tensor(self.feature_text_indexes),  # shape: word count
            # torch.tensor(self.feature_parent_text_indexes),  # shape: word count
            torch.tensor(self.feature_character_indexes),  # shape: word count * max_word_length
            # torch.tensor(self.feature_parent_character_indexes),  # shape: word count * max_word_length
            torch.tensor(self.tag),
            torch.tensor(self.node_string_types, dtype=torch.float),
            torch.tensor(self.feature_location, dtype=torch.long),
            torch.tensor(self.feature_xpath, dtype=torch.long),
            torch.tensor(self.feature_class_probabilities),
            torch.tensor(self.feature_node_embedding),
            torch.tensor(self.feature_common_features),
        )


@pytest.fixture
def local_module_hyper_parameters():

    return LocalModuleHyperParameters(
        character_embedding_dim=10,
        word_embedding_dim=4,
        cnn_character_embedding_kernel_size=3,
        cnn_character_embedding_filter_count=5,
        rnn_hidden_dimension=10,
        discrete_features_embedding_dim=5,
        common_features_embedding_dim=4,
        dropout_rate=0,
    )


@pytest.fixture
def relational_module_hyper_parameters(model_temp_path):

    return RelationalModuleHyperParameters(
        xpath_embedding_dim=10,
        position_embedding_dim=5,
        xpath_lstm_hidden_dim=10,
        position_size=10,
        m_top_uncertain_field_nodes=3,
        label_voting_threshold=1,
        local_node_embedding_model_path=str(model_temp_path / "model.pt"),
        dropout_rate=0,
    )


@pytest.fixture
def vocabulary_indexer():
    return {"<UNK>": 0, "book": 1, "online": 2, "browse": 3, "policy": 4}


@pytest.fixture
def cvi():
    return {
        "<UNK>": 0,
        "b": 1,
        "o": 2,
        "k": 3,
        "n": 4,
        "l": 5,
        "i": 6,
        "e": 7,
        "r": 8,
        "w": 9,
        "s": 10,
        "p": 11,
        "c": 12,
        "y": 13,
    }


@pytest.fixture
def mwl():
    return 6


@pytest.fixture
def word_embedding_model():
    return {
        "book": [0, 0, 0, 1],
        "online": [0, 0, 1, 0],
        "browse": [0, 1, 0, 0],
        "policy": [1, 0, 0, 0],
        "<UNK>": [0, 0, 0, 0],
    }, 4


@pytest.fixture
def data_tree_example(word_embedding_model, cvi, mwl):
    pretrained_word_embeddings, embedding_dimension = word_embedding_model
    a = FreeDOMDataTree(
        "a",
        FreeDOMFeatureVector(
            feature_text_indexes=FreeDOMFeatures.build_english_text_tensor(
                "book", mwl, 20, pretrained_word_embeddings, embedding_dimension
            ),
            feature_parent_text_indexes=FreeDOMFeatures.build_english_text_tensor(
                "", mwl, 20, pretrained_word_embeddings, embedding_dimension
            ),
            feature_character_indexes=FreeDOMFeatures.build_text_character_tensor("book", cvi, mwl, 20),
            feature_parent_character_indexes=FreeDOMFeatures.build_text_character_tensor("", cvi, mwl, 20),
            tag=[TAG_TO_INT["h1"]],
            node_string_types=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            feature_location=5,
            feature_xpath=[TAG_TO_INT[tag] for tag in ["body", "div", "h1"]],
            feature_node_embedding=np.random.rand(54).tolist(),
            feature_class_probabilities=[0.2, 0.92, 0.4, 0.2],
            feature_tentative_label="a",
            feature_common_features=[0, 0, 0, 1],
        ),
        element_index=0,
    )  # pylint: disable=not-callable
    b = FreeDOMDataTree(
        "b",
        FreeDOMFeatureVector(
            feature_text_indexes=FreeDOMFeatures.build_english_text_tensor(
                "online", mwl, 20, pretrained_word_embeddings, embedding_dimension
            ),
            feature_parent_text_indexes=FreeDOMFeatures.build_english_text_tensor(
                "book", mwl, 20, pretrained_word_embeddings, embedding_dimension
            ),
            feature_character_indexes=FreeDOMFeatures.build_text_character_tensor("online", cvi, mwl, 20),
            feature_parent_character_indexes=FreeDOMFeatures.build_text_character_tensor("book", cvi, mwl, 20),
            tag=[TAG_TO_INT["img"]],
            node_string_types=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            feature_location=5,
            feature_xpath=[TAG_TO_INT[tag] for tag in ["body", "div", "img"]],
            feature_node_embedding=np.random.rand(54).tolist(),
            feature_class_probabilities=[0.04, 0.02, 0.92, 0.02],
            feature_tentative_label="b",
            feature_common_features=[0, 0, 0, 1],
        ),
        element_index=1,
    )
    c = FreeDOMDataTree(
        "c",
        FreeDOMFeatureVector(
            feature_text_indexes=FreeDOMFeatures.build_english_text_tensor(
                "browse", mwl, 20, pretrained_word_embeddings, embedding_dimension
            ),
            feature_parent_text_indexes=FreeDOMFeatures.build_english_text_tensor(
                "online", mwl, 20, pretrained_word_embeddings, embedding_dimension
            ),
            feature_character_indexes=FreeDOMFeatures.build_text_character_tensor("browse", cvi, mwl, 20),
            feature_parent_character_indexes=FreeDOMFeatures.build_text_character_tensor("browse", cvi, mwl, 20),
            tag=[TAG_TO_INT["span"]],
            node_string_types=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            feature_location=5,
            feature_xpath=[TAG_TO_INT[tag] for tag in ["body", "div", "span"]],
            feature_node_embedding=np.random.rand(54).tolist(),
            feature_class_probabilities=[0.92, 0.02, 0.04, 0.02],
            feature_tentative_label=SimpleStageOneClassifier.DEFAULT_LABEL,
            feature_common_features=[0, 0, 0, 1],
        ),
        element_index=2,
    )
    d = FreeDOMDataTree(
        "unlabeled",
        FreeDOMFeatureVector(
            feature_text_indexes=FreeDOMFeatures.build_english_text_tensor(
                "policy", mwl, 20, pretrained_word_embeddings, embedding_dimension
            ),
            feature_parent_text_indexes=FreeDOMFeatures.build_english_text_tensor(
                "browse", mwl, 20, pretrained_word_embeddings, embedding_dimension
            ),
            feature_character_indexes=FreeDOMFeatures.build_text_character_tensor("policy", cvi, mwl, 20),
            feature_parent_character_indexes=FreeDOMFeatures.build_text_character_tensor("browse", cvi, mwl, 20),
            tag=[TAG_TO_INT["title"]],
            node_string_types=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            feature_location=7,
            feature_xpath=[TAG_TO_INT[tag] for tag in ["body", "div", "span", "title"]],
            feature_node_embedding=np.random.rand(54).tolist(),
            feature_class_probabilities=[0.92, 0.02, 0.02, 0.04],
            feature_tentative_label=SimpleStageOneClassifier.DEFAULT_LABEL,
            feature_common_features=[0, 0, 0, 1],
        ),
        element_index=3,
    )

    a.add_child(b)
    a.add_child(c)
    b.add_child(d)

    return [a, b, c, d]


def test_freedom_save_load(model_temp_path, local_module_hyper_parameters):
    path = model_temp_path

    mdl = FreeDOMClassifier(
        4,
        4,
        latent_dim=14,
        local_module_hyper_parameters=local_module_hyper_parameters,
        languages=["en", "de", "no", "nl", "se", "fi"],
        pretrained_word_embedding_model={},
        word_vocabulary=["test"],
        character_vocabulary=["t", "e", "s", "t"],
        max_word_length=4,
    )

    mdl.save(loc=path)

    loaded_mdl = FreeDOMClassifier.load(loc=path)

    assert loaded_mdl is not mdl
    assert isinstance(loaded_mdl, FreeDOMClassifier)
    for par1, par2 in zip_longest(mdl.parameters(), loaded_mdl.parameters()):
        assert torch.equal(par1, par2)  # pylint: disable=no-member


def test_freedom_overfit(
    local_module_hyper_parameters,
    vocabulary_indexer,
    cvi,
    mwl,
    word_embedding_model,
    data_tree_example,
    model_temp_path,
):
    path = model_temp_path
    torch.manual_seed(1337)

    a, b, c, d = data_tree_example

    model = SimpleStageOneClassifier(
        4,
        4,
        latent_dim=14,
        local_module_hyper_parameters=local_module_hyper_parameters,
        word_vocabulary=vocabulary_indexer,
        max_word_length=mwl,
        character_vocabulary=cvi,
        pretrained_word_embedding_model=word_embedding_model,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = SimpleTrainer(model, optimizer, loss_fn)

    trainer.train(DataLoader([a, b, c, d], batch_size=4, collate_fn=list, num_workers=0), 100)

    assert model.predict(a) == "a"
    assert model.predict(b) == "b"
    assert model.predict(c) == "c"
    assert model.predict(d) == "unlabeled"

    model.save(loc=path)


def test_freedom_stage2_label_split(relational_module_hyper_parameters, data_tree_example):
    model = SimpleStageTwoClassifier(
        4, 4, latent_dim=14, relational_module_hyper_parameters=relational_module_hyper_parameters
    )
    assert ("none", "value") == model.split_label("none-value")
    assert ("value", "none") == model.split_label("value-none")
    assert ("none", "none") == model.split_label("none-none")
    assert ("value", "value") == model.split_label("value-value")

    a, b, c, _ = data_tree_example
    assert len(model.predict(model._create_pair(c, c), splitted=True)) == 2
    assert model.predict(model._create_pair(c, a), splitted=True)[0] in ["none", "value"]
    assert model.predict(model._create_pair(c, a), splitted=True)[1] in ["none", "value"]
    assert isinstance(model.predict(model._create_pair(c, b)), str)


def test_freedom_stage2_certain_uncertain(relational_module_hyper_parameters, data_tree_example):
    a, _, _, _ = data_tree_example
    model = SimpleStageTwoClassifier(
        4, 4, latent_dim=14, relational_module_hyper_parameters=relational_module_hyper_parameters
    )

    certain_fields, uncertain_fields = model.get_certain_uncertain_fields(a)
    assert "c" in uncertain_fields
    assert "a" in certain_fields
    assert "b" in certain_fields

    assert len(certain_fields) == 2
    assert len(uncertain_fields) == 1
    assert len(certain_fields["a"]) == 1
    assert len(certain_fields["b"]) == 1
    assert len(uncertain_fields["c"]) == 2
    assert SimpleStageTwoClassifier.DEFAULT_LABEL not in uncertain_fields
    assert SimpleStageTwoClassifier.DEFAULT_LABEL not in certain_fields


def test_freedom_stage2_candidate_pairs(relational_module_hyper_parameters, data_tree_example):
    a, _, _, _ = data_tree_example

    model = SimpleStageTwoClassifier(
        4, 4, latent_dim=14, relational_module_hyper_parameters=relational_module_hyper_parameters
    )

    candidate_nodes, pairs_indices, uncertain_mask, certain_fields, uncertain_fields = model.get_candidate_pairs(a)

    assert "c" in uncertain_fields
    assert "a" in certain_fields
    assert "b" in certain_fields

    assert len(candidate_nodes) == 4
    assert len(pairs_indices) == 12

    assert sum([1 if item else 0 for item in uncertain_mask]) == 2
    assert uncertain_mask == [False, False, True, True]


def test_freedom_stage2_overfit(relational_module_hyper_parameters, data_tree_example):
    a, b, c, d = data_tree_example

    model = SimpleStageTwoClassifier(
        4, 4, latent_dim=14, relational_module_hyper_parameters=relational_module_hyper_parameters
    )

    pair_list = []
    for head_node in data_tree_example:
        for tail_node in data_tree_example:
            pair_list.append(model._create_pair(head_node, tail_node))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = SimpleTrainer(model, optimizer, loss_fn)
    trainer.train(DataLoader(pair_list, batch_size=1, collate_fn=list, num_workers=0), 100)

    assert model.predict(model._create_pair(a, a)) == "value-value"
    assert model.predict(model._create_pair(a, b)) == "value-value"
    assert model.predict(model._create_pair(a, c)) == "value-value"
    assert model.predict(model._create_pair(a, d)) == "value-none"

    assert model.predict(model._create_pair(b, a)) == "value-value"
    assert model.predict(model._create_pair(b, b)) == "value-value"
    assert model.predict(model._create_pair(b, c)) == "value-value"
    assert model.predict(model._create_pair(b, d)) == "value-none"

    assert model.predict(model._create_pair(c, a)) == "value-value"
    assert model.predict(model._create_pair(c, b)) == "value-value"
    assert model.predict(model._create_pair(c, c)) == "value-value"
    assert model.predict(model._create_pair(c, d)) == "value-none"

    assert model.predict(model._create_pair(d, a)) == "none-value"
    assert model.predict(model._create_pair(d, b)) == "none-value"
    assert model.predict(model._create_pair(d, c)) == "none-value"
    assert model.predict(model._create_pair(d, d)) == "none-none"

    candidate_nodes, pairs_indices, _, _, _ = model.get_candidate_pairs(a)

    votes = model._count_votes(candidate_nodes, pairs_indices)

    assert len(votes) == 4
    assert votes == [6, 6, 0, 6]

    tree_predictions = model.predict_tree(a)
    assert all(node.label == "a" for prob, node in tree_predictions["a"])
    assert all(node.label == "b" for prob, node in tree_predictions["b"])
    assert all(node.label == "c" for prob, node in tree_predictions["c"])
    assert all(node.label == "unlabeled" for prob, node in tree_predictions["unlabeled"])

    assert (
        len(tree_predictions["a"])
        == len(tree_predictions["b"])
        == len(tree_predictions["c"])
        == len(tree_predictions["unlabeled"])
        == 1
    )
