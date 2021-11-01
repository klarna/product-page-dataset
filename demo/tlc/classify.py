# pylint: disable-all
# type: ignore
"""
Entry point for tree-lstm element classifier
"""

# TODO: fix this file to work with new trees

from argparse import ArgumentParser

import webtraversallibrary as wtf
from webtraversallibrary.classifiers import ElementClassifier
from webtraversallibrary.policies import DUMMY

from .dataset.utilities import LABELS_OF_INTEREST
from .models.prototypes import TreeClassifier
from .structures.trees import DataTree

CLASS_COLORS = [
    wtf.Color(255, 0, 0),
    wtf.Color(0, 255, 0),
    wtf.Color(0, 0, 255),
    wtf.Color(255, 255, 0),
    wtf.Color(255, 0, 255),
    wtf.Color(255, 255, 255),
]


def main(url, model_loc, cutoff):
    tree_model = TreeClassifier.load(model_loc)
    tree_model.eval()

    def _classifier_callback(elements, _):
        tree = DataTree.from_elements(elements)

        labeled_nodes = dict()
        for node in tree:
            class_probs = tree_model.softmax(tree_model(node)).tolist()

            for idx, prob in enumerate(class_probs):
                if idx > 0:
                    nm = LABELS_OF_INTEREST[idx - 1]
                    labeled_nodes.setdefault(nm, [])
                    labeled_nodes[nm].append((node.element, prob))

        return labeled_nodes

    tree_classifier = ElementClassifier(
        name="tlc", callback=_classifier_callback, highlight=cutoff, highlight_color=wtf.Color(255, 0, 0)
    )

    wf = wtf.Workflow(url=url, policy=DUMMY, config=wtf.Config(["default"]), classifiers=[tree_classifier])

    wf.run_once()

    while not input("Quit? (y/N)").lower().startswith("y"):
        continue

    wf.quit()


if __name__ == "__main__":

    parser = ArgumentParser(description="Classify the elements on a live website")

    parser.add_argument("--url", "-u", required=True, help="Url of the product page.")
    parser.add_argument("--model-loc", "-m", required=True, help="Location of the model to use for classification")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--show-top", "-t", type=int, default=1, dest="cutoff", help="Show the n top scoring elements for each class."
    )
    group.add_argument(
        "--min-conf", "-p", type=float, dest="cutoff", default=0.8, help="Lower bound on acceptable class confidence."
    )

    main(**vars(parser.parse_args()))
