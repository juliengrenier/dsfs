import math
from collections import Counter
from typing import List

from .vector import sum_of_squares, dot_product


def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs)

def median(xs: List[float]) -> float:
    sorted_xs = sorted(xs)
    length = len(sorted_xs)
    midpoint = length // 2
    if length % 2 == 0:
        return (sorted_xs[midpoint-1] + sorted_xs+1) /2
    else:
        return sorted_xs[midpoint]


def quantile(xs: List[float], p: float) -> float:
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def mode(xs: List[float]) -> List[float]:
    counts = Counter(xs)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)


def de_mean(xs: List[float]) -> float:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    n = len(xs)
    assert n >= 2, "need more than 2 elements"
    deviation = de_mean(xs)
    return sum_of_squares(deviation) / (n-1)


def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))


def interquartile_range(xs: List[float], p1: float = 0.25, p2: float = 0.75) -> float:
    return quantile(xs, p=p2) - quantile(xs, p=p1)


def covariance(xs: List[float], ys: List[float]) -> float:
    return dot_product(de_mean(xs), de_mean(ys)) / (len(xs) -1)


def correlation(xs: List[float], ys: List[float]) -> float:
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    return 0
