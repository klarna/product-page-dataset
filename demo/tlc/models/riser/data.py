"""
Handles data processing for RiSER
"""
from collections import deque
from itertools import islice
from typing import Iterator, List, Set, Tuple

import bs4
from spacy.tokens import Token

# Special tokens for vocabulary
from tlc.dataset.singletons import get_spacy_model

PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
URL = "<URL>"
ALNUM = "<ALNUM>"
EOS = "<EOS>"

# Element types to skip when traversing DOM tree
SKIP = ["script", "style"]


class Vocabulary:
    """
    Dictionary of token to index in vocabulary
    """

    def __init__(self, training_vocab: Set[str]):
        training_vocab.update({UNK, PAD})
        self.vocab = {token: i for i, token in enumerate(sorted(training_vocab))}

    def __getitem__(self, item: str) -> int:
        return self.vocab.get(item, self.vocab[UNK])

    def __len__(self) -> int:
        return len(self.vocab)


class Document:
    """
    TODO Refactor this class
    Serializes document from string of HTML document
    """

    def __init__(self, root: bs4.BeautifulSoup, max_doc_len: int, max_xpath_len: int):
        self.max_doc_len = max_doc_len
        self.max_xpath_len = max_xpath_len

        html = list(islice(Document.traverse_inorder(root, []), self.max_doc_len))
        if not html:
            self.words = None
            self.paths = None
        else:
            words, paths = zip(*html)
            self.words = self.pad_words(list(words))
            self.paths = self.clip_and_pad_paths(list(paths))

    def pad_words(self, words: List[str]) -> List[str]:
        words = (words + self.max_doc_len * [PAD])[: self.max_doc_len]
        return words

    def clip_and_pad_paths(self, paths: List[List[str]]) -> List[List[str]]:
        """
        Normalize document so that each MAX_DOC_LEN token has an associated XPath sequence of
        length MAX_XPATH_LEN (if token does not exist then assc. XPath consists of all PAD tokens)
        :param paths:
        :return:
        """
        paths = [list(deque(path, maxlen=self.max_xpath_len)) for path in paths]
        paths = [(path + self.max_xpath_len * [PAD])[: self.max_xpath_len] for path in paths]
        paths = (paths + self.max_doc_len * [self.max_xpath_len * [PAD]])[: self.max_doc_len]
        return paths

    @staticmethod
    def traverse_inorder(root: bs4.PageElement, path: List) -> Iterator[Tuple[str, List[str]]]:
        """
        Performs recursive in-order traversal of bs4 tree
        :param root:
        :param path:
        :return:
        """
        # Check whether the current node is a tag or inner text. If text stop recursing.
        if isinstance(root, bs4.NavigableString):
            tokens = Document.tokenize_and_format_text(root.string)
            for token in tokens:
                yield (token, path)
        elif isinstance(root, bs4.Tag) and root.name not in SKIP:
            for subtree in root.children:
                yield from Document.traverse_inorder(subtree, path + [root.name])

    @staticmethod
    def tokenize_and_format_text(raw_text: str) -> List[str]:
        """
        Tokenize raw string obtained from bs4 page element and apply some formatting
        (lowercase, mask numbers/alphanum sequences etc)
        :param raw_text: Attribute 'string' from bs4.NavigableString type, raw string from HTML element
        :return: List of processed tokens of text from the HTML element
        """

        def is_alnum(tok: Token) -> bool:
            return any(c.isalpha() for c in tok.text) and any(c.isdigit() for c in tok.text)

        SP = get_spacy_model()
        spacy_tokens = SP(raw_text.strip())

        formatted_tokens = []
        for tok in spacy_tokens:
            if tok.is_space or tok.is_stop:
                continue
            if tok.like_num:
                formatted_tokens.append(NUM)
            elif tok.like_url:
                formatted_tokens.append(URL)
            elif is_alnum(tok):
                formatted_tokens.append(ALNUM)
            else:
                formatted_tokens.append(tok.text.lower())

        if formatted_tokens:
            formatted_tokens.append(EOS)

        return formatted_tokens
