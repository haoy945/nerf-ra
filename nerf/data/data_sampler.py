import torch
from torch.utils.data.sampler import Sampler


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices.
    """

    def __init__(self, size: int, shuffle: bool = True):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
        """
        if not isinstance(size, int):
            raise TypeError(f"TrainingSampler(size=) expects an int. Got type {type(size)}.")
        if size <= 0:
            raise ValueError(f"TrainingSampler(size=) expects a positive int. Got {size}.")
        self._size = size
        self._shuffle = shuffle

    def __iter__(self):
        yield from self._infinite_indices()

    def _infinite_indices(self):
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size).tolist()
            else:
                yield from torch.arange(self._size).tolist()


class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    """

    def __init__(self, size: int) -> None:
        if not isinstance(size, int):
            raise TypeError(f"TrainingSampler(size=) expects an int. Got type {type(size)}.")
        if size <= 0:
            raise ValueError(f"TrainingSampler(size=) expects a positive int. Got {size}.")
        self._size = size
        self._indices = range(size)

    def __iter__(self):
        yield from self._indices

    def __len__(self):
        return self._size
