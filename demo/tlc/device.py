"""
Module that act as a singleton to access device to use.
"""

#  pylint: # pylint: disable=global-statement
import gin
import torch

torch_device = None


@gin.configurable()
def get_torch_device(use_gpu: bool = False):
    """
    Singleton to return the torch device to use.

    :param use_gpu: Flag to indicate if gpu should be used.
    :return: Torch device.
    """
    global torch_device
    if torch_device is None:
        torch_device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    return torch_device


def reset_device():
    """
    Method to delete the global device variable.
    Used by the augmentation script.
    :return:
    """
    global torch_device
    torch_device = None
