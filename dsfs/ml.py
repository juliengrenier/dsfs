import random

from typing import TypeVar, List, Tuple

X = TypeVar('X')
Y = TypeVar('Y')


def split_data(data: List[X], prob: float, shuffle: bool = True) -> Tuple[List[X], List[X]]:
    data = data[:]
    if shuffle: random.shuffle(data)

    cut = int(len(data) * prob)
    return data[:cut], data[cut:]


def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idx = [i for i in range(len(xs))]
    train_idx, test_idx = split_data(idx, 1 - test_pct)
    return (
        [xs[i] for i in train_idx],
        [xs[i] for i in test_idx],
        [ys[i] for i in train_idx],
        [ys[i] for i in test_idx]
    )
