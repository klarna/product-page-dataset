from pathlib import Path

import pytest

from tlc.dataset.vocabulary.vocabulary import Vocabulary

parent_path = Path(__file__).resolve().parents[0]


def test_vocabulary_construction():
    token_list = ["first", "second", "third", "fourth", "fifth"]
    vocabulary = Vocabulary(tokens=token_list)

    assert vocabulary.token_count() == 5
    assert vocabulary.get_tokens() == token_list
    assert not vocabulary.contains_defaults
    assert vocabulary.metadata == {}
    assert vocabulary.token_to_index["second"] == 1
    assert vocabulary.index_to_token[1] == "second"


def test_token_count():
    token_list = ["first", "second", "third", "fourth", "fifth"]
    vocabulary = Vocabulary(tokens=token_list)

    assert vocabulary.token_count() == 5
    assert vocabulary.token_count(contain_defaults=True) == 5

    assert vocabulary.token_count() == 5
    assert vocabulary.token_count(contain_defaults=True) == 5

    vocabulary.add_token("first")
    assert vocabulary.token_count() == 5

    vocabulary.add_token("sixth")
    assert vocabulary.token_count() == 6

    token_list = ["first", "second", "third", "fourth", "fifth"]
    vocabulary = Vocabulary(tokens=token_list, contain_defaults=True)

    assert vocabulary.token_count() == 5
    assert vocabulary.token_count(contain_defaults=True) == 5 + len(Vocabulary.default_tokens)
    assert vocabulary.token_count(contain_defaults=False) == 5


def test_add_token():
    token_list = ["first", "second", "third", "fourth", "fifth"]
    vocabulary = Vocabulary(tokens=token_list)

    vocabulary.add_token("first")
    assert vocabulary.index_to_token[0] == "first"
    assert vocabulary.token_to_index["first"] == 0

    vocabulary.add_token("sixth")
    assert vocabulary.index_to_token[5] == "sixth"
    assert vocabulary.token_to_index["sixth"] == 5

    vocabulary.add_tokens(["seventh", "eighth"])
    assert vocabulary.index_to_token[6] == "seventh"
    assert vocabulary.token_to_index["eighth"] == 7

    token_list = ["first", "second", "third", "fourth", "fifth"]
    vocabulary = Vocabulary(tokens=token_list, contain_defaults=True)

    assert vocabulary.token_to_index["first"] == len(Vocabulary.default_tokens)
    assert vocabulary.index_to_token[0] == Vocabulary.default_tokens[0]
    assert vocabulary.token_to_index[Vocabulary.default_tokens[0]] == 0


def test_from_json():
    token_list = ["first", "second", "third", "fourth", "fifth"]

    vocabulary_json = {"tokens": token_list, "contains_defaults": False, "metadata": {}}

    vocabulary = Vocabulary.from_json(vocabulary_json)
    assert vocabulary.token_count() == 5
    assert vocabulary.get_tokens() == token_list
    assert not vocabulary.contains_defaults
    assert vocabulary.metadata == {}
    assert vocabulary.token_to_index["second"] == 1
    assert vocabulary.index_to_token[1] == "second"

    vocabulary_json = {"tokens": token_list, "metadata": {}}

    with pytest.raises(KeyError):
        Vocabulary.from_json(vocabulary_json)


def test_to_json():
    token_list = ["first", "second", "third", "fourth", "fifth"]
    vocabulary = Vocabulary(tokens=token_list)

    vocabulary_json = vocabulary.to_json()
    assert vocabulary_json["token_count"] == 5
    assert vocabulary_json["tokens"] == token_list
    assert not vocabulary_json["contains_defaults"]
    assert vocabulary_json["metadata"] == {}
    assert vocabulary_json["tokens"][1] == "second"


def test_to_json_file(tmp_path):
    token_list = ["first", "second", "third", "fourth", "fifth"]
    vocabulary = Vocabulary(tokens=token_list)

    vocabulary.to_json_file(json_file_path=tmp_path / "test_vocabulary.json")
    reloaded_vocabulary = Vocabulary.from_json_file(json_file_path=tmp_path / "test_vocabulary.json")

    assert vocabulary.token_count() == reloaded_vocabulary.token_count()
    assert vocabulary.get_tokens() == reloaded_vocabulary.get_tokens()
    assert not reloaded_vocabulary.contains_defaults
    assert vocabulary.metadata == reloaded_vocabulary.metadata
    assert vocabulary.token_to_index["second"] == reloaded_vocabulary.token_to_index["second"]
    assert vocabulary.index_to_token[1] == reloaded_vocabulary.index_to_token[1]


def test_from_json_file():
    vocabulary = Vocabulary.from_json_file(parent_path / Path("test_files/test_vocabulary.json"))

    assert vocabulary.token_count() == 5
    assert vocabulary.get_tokens() == ["first", "second", "third", "fourth", "fifth"]
    assert not vocabulary.contains_defaults
    assert vocabulary.metadata == {}
    assert vocabulary.token_to_index["second"] == 1
    assert vocabulary.index_to_token[1] == "second"
