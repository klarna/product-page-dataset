"""
Module containing Mixin classes used in the dataset.
"""

import json
import pickle
from abc import abstractmethod
from pathlib import Path


class JsonMixin:
    """
    Mixin to add save and load using json to a class.
    """

    @abstractmethod
    def to_json(self):
        """
        Return json representation of the object.
        Classes which inherit this Mixin should implement this class

        :return: Json representation of object.
        """

    @classmethod
    def from_json(cls, object_json):
        """
        Method to load object of class from a json.

        :param object_json: Json holding the vocabulary information.
        :return: Class object loaded with the json information.
        """

    def to_json_file(self, json_file_path: Path):
        """
        Save object as json to file.

        :param json_file_path: The path of the file to save the json.
        """
        with json_file_path.open(mode="w") as json_file:
            json.dump(self.to_json(), json_file, indent=4, sort_keys=True)

    @classmethod
    def from_json_file(cls, json_file_path: Path):
        """
        Method to load class object from a json file.

        :param json_file_path: Path to the json file holding the objects json representation.
        :return: Class object loaded with the json.
        """
        with json_file_path.open(mode="r") as json_file:
            object_json = json.load(json_file)

        return cls.from_json(object_json)


class PickleMixin:
    """
    Mixin to add pickling and unpickling functionality to class.
    """

    def to_pickle_file(self, pickle_file_path: Path):
        """
        Save vocabulary as pickle to file.

        :param pickle_file_path: The path of the file to save the pickle.
        """
        with pickle_file_path.open(mode="wb") as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def from_pickle_file(cls, pickle_file_path: Path):
        """
        Method to load class object from a json file.

        :param pickle_file_path: Path to the pickle file.
        :return: Class object.
        """
        with pickle_file_path.open(mode="rb") as pickle_file:
            new_object = pickle.load(pickle_file)

        return new_object
