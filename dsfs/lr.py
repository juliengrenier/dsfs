from typing import Tuple

from dsfs.vector import Vector
from dsfs.stats import correlation, standard_deviation, mean, de_mean


def predict(alpha: float, beta: float, x: float) -> float:
    return beta * x + alpha


def error(alpha: float, beta: float, x: float, y: float) -> float:
    return predict(alpha, beta, x) - y


def sum_of_sqerror(alpha: float, beta:float, xs: Vector, ys: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i)**2 for x_i, y_i in zip(xs, ys))


def least_squares_fit(xs: Vector, ys: Vector) -> Tuple[float, float]:
    beta = correlation(xs, ys) * standard_deviation(ys) / standard_deviation(xs)
    alpha = mean(ys) - beta * mean(xs)
    return alpha, beta


def total_sum_of_squares(xs: Vector) -> float:
    return sum(x ** 2 for x in de_mean(xs))

def r_squared(alpla: float, beta: float, xs: Vector, ys: Vector) -> float:
    return 1 - (sum_of_sqerror(alpha, beta, xs, ys) / total_sum_of_squares(ys))
