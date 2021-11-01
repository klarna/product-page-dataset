"""
Module that contains decorators used in the project.
"""


def add_options(options):
    """
    Decorator to convert a list of click options to @click.options.
    :param options: List of Click options.
    :return:
    """

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options
