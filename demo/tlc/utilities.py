"""
Various utilities used in the project
"""

import logging
import random
from functools import wraps
from itertools import groupby
from typing import Any, Callable, Iterable, List, Optional


def local_cache(method: Callable) -> Callable:
    """Implements an object local lru cache; the wrapped method cannot have arguments so use only for properties
    and the like"""
    # pylint: disable=protected-access

    @wraps(method)
    def _f(self: Any) -> Any:
        if not hasattr(self, "__local_cache"):
            self.__local_cache = dict()

        if method.__name__ not in self.__local_cache:
            self.__local_cache[method.__name__] = method(self)

        return self.__local_cache[method.__name__]

    return _f


def stream_sample(stream: Iterable, k: int) -> List[Any]:
    """Sample k elements (without replacement) with equal probability from the stream"""

    idx = 0
    reservoir = [None] * k
    for element in stream:
        if idx < k:
            replace = idx
        else:
            replace = random.randrange(idx + 1)
        if replace < k:
            reservoir[replace] = element
        idx += 1
    return [x for x in reservoir if x is not None]


def all_equal(iterable: Iterable[Any]) -> bool:
    """Returns true if all the elements in the iterable are equal"""
    g = groupby(iterable)
    return bool(next(g, True)) and not bool(next(g, False))


def setup_logging(logging_level: Optional[int] = logging.DEBUG):
    """
    Sets up logging: create a directory to write log files to, configure handlers. Sets sane
    default values for in-house and third-party modules.

    :param logging_level: level of logging you wish to have, accepts number or logging.LEVEL
    """
    formatter = logging.Formatter("{asctime} {levelname:8} {name:12} - {message}", "%H:%M:%S", style="{")

    to_console = logging.StreamHandler()
    to_console.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging_level)

    selenium_logger = logging.getLogger("selenium")
    selenium_logger.setLevel(logging.INFO)

    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)

    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.INFO)

    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.WARNING)

    sh_logger = logging.getLogger("sh")
    sh_logger.setLevel(logging.WARNING)

    boto_logger = logging.getLogger("botocore")
    boto_logger.setLevel(logging.WARNING)

    handlers = list(logger.handlers)
    for handler in handlers:
        logger.removeHandler(handler)

    logger.addHandler(to_console)
