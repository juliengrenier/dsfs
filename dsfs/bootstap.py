import random
from typing import TypeVar, Callable


X = TypeVar('X')
Stat = TypeVar('Stat')


def bootstrap_sample(data: List[X]) -> List[X]:
    """ Returns a list of len(data) samples from data with replacement """
    return [random.choice(data) for _ in data]


def bootstrap_stastistic(data: List[X], stats_fn: Callable[[List[X]], Stat], num_samples: int) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

