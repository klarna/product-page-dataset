# pylint: disable=redefined-outer-name
from pathlib import Path

import pytest

from tlc.dataset.vocabulary.multilanguagevocabulary import MultiLanguageVocabulary
from tlc.dataset.vocabulary.vocabulary import Vocabulary

parent_path = Path(__file__).resolve().parents[0]


@pytest.fixture
def multi_language_vocabulary():
    token_list = ["first", "second", "third", "fourth", "fifth"]

    token_map = {
        "first": Vocabulary(tokens=token_list),
        "second": Vocabulary(tokens=token_list),
        "third": Vocabulary(tokens=token_list),
        "fourth": Vocabulary(tokens=token_list),
        "fifth": Vocabulary(tokens=token_list),
    }

    return MultiLanguageVocabulary(tokens_map=token_map)


@pytest.fixture
def vocabulary_json():
    token_list = ["first", "second", "third", "fourth", "fifth"]

    return {"tokens": token_list, "contains_defaults": False, "metadata": {}}


@pytest.fixture
def multi_language_vocabulary_json(vocabulary_json):
    return {
        "vocabularies": {
            "first": vocabulary_json.copy(),
            "second": vocabulary_json.copy(),
            "third": vocabulary_json.copy(),
            "fourth": vocabulary_json.copy(),
            "fifth": vocabulary_json.copy(),
        },
        "languages": ["first", "second", "third", "fourth", "fifth"],
        "language_count": 5,
        "token_count": 25,
        "metadata": {},
    }


def test_vocabulary_construction(multi_language_vocabulary):
    assert multi_language_vocabulary.total_token_count == 5 * 5
    assert multi_language_vocabulary.language_count == 5
    assert multi_language_vocabulary.metadata == {}
    assert multi_language_vocabulary.get_vocabulary("second").token_count() == 5
    with pytest.raises(KeyError):
        multi_language_vocabulary.get_vocabulary("sixth")


def test_add_token(multi_language_vocabulary):

    multi_language_vocabulary.add_token(language="first", token="sixth")
    assert multi_language_vocabulary.get_vocabulary("first").index_to_token[5] == "sixth"

    multi_language_vocabulary.add_token(language="sixth", token="first")
    assert multi_language_vocabulary.language_count == 6
    assert multi_language_vocabulary.total_token_count == 5 * 5 + 1 + 1

    multi_language_vocabulary.add_tokens(language="seventh", tokens=["first", "second"])
    assert multi_language_vocabulary.language_count == 7
    assert multi_language_vocabulary.total_token_count == 5 * 5 + 1 + 1 + 2


def test_from_json(multi_language_vocabulary_json):
    token_list = ["first", "second", "third", "fourth", "fifth"]

    multi_language_vocabulary = MultiLanguageVocabulary.from_json(multi_language_vocabulary_json)
    assert multi_language_vocabulary.language_count == 5
    assert multi_language_vocabulary.languages == token_list
    assert multi_language_vocabulary.metadata == {}
    assert multi_language_vocabulary.get_vocabulary("first").token_count() == 5

    vocabulary_json = {"tokens": token_list, "metadata": {}}

    with pytest.raises(KeyError):
        Vocabulary.from_json(vocabulary_json)


def test_to_json(multi_language_vocabulary):
    token_list = ["first", "second", "third", "fourth", "fifth"]

    vocabulary_json = multi_language_vocabulary.to_json()
    assert vocabulary_json["language_count"] == 5
    assert not vocabulary_json["vocabularies"]["third"]["contains_defaults"]
    assert vocabulary_json["metadata"] == {}
    assert vocabulary_json["languages"] == token_list
    assert vocabulary_json["token_count"] == 5 * 5


def test_to_json_file(tmp_path, multi_language_vocabulary):

    multi_language_vocabulary.to_json_file(json_file_path=tmp_path / "test_vocabulary.json")
    reloaded_vocabulary = MultiLanguageVocabulary.from_json_file(json_file_path=tmp_path / "test_vocabulary.json")

    assert multi_language_vocabulary.total_token_count == reloaded_vocabulary.total_token_count
    assert multi_language_vocabulary.language_count == reloaded_vocabulary.language_count
    assert sorted(multi_language_vocabulary.languages) == reloaded_vocabulary.languages
    assert multi_language_vocabulary.metadata == reloaded_vocabulary.metadata


def test_from_json_file():
    multi_language_vocabulary = MultiLanguageVocabulary.from_json_file(
        parent_path / Path("test_files/test_multi_language_vocabulary.json")
    )

    assert multi_language_vocabulary.total_token_count == 5 * 5
    assert multi_language_vocabulary.languages == ["first", "second", "third", "fourth", "fifth"]
    assert multi_language_vocabulary.language_count == 5
    assert not multi_language_vocabulary.get_vocabulary("first").contains_defaults
    assert multi_language_vocabulary.metadata == {}
