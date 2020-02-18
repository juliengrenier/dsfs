from typing import Tuple, List
import random
import tqdm

from dsfs.vector import Vector, dot_product
from dsfs.matrix import Matrix, vector_mean
from dsfs.gradients import gradient_step
from dsfs.lr import total_sum_of_squares


def predict(xs: Vector, beta: Vector) -> float:
    assert xs[0] == 1, "The first element of xs must be 1"
    return dot_product(xs, beta)


def error(xs: Vector, y: float, beta: Vector) -> float:
    return predict(xs, beta) - y


def squared_error(xs: Vector, y: float, beta: Vector) -> float:
    return error(xs, y, beta) ** 2

xs = [1,2,3]
y = 30
beta =[4,4,4]

assert predict(xs, beta) == sum([4,8,12])
assert error(xs, y, beta ) == -6
assert squared_error(xs, y, beta) == 36


def sqerror_gradient(xs: Vector, y:float, beta: Vector) -> Vector:
    err = error(xs, y, beta)
    return [2 * err * x_i for x_i in xs]

assert sqerror_gradient(xs, y, beta) == [-12, -24, -36]

def least_squares_fit(
        xs: Matrix,
        ys: Vector,
        learning_rate: float = 0.001,
        num_steps: int = 1000,
        batch_size:int = 1
):
    guess = [random.random() for _ in xs[0]]
    for _ in tqdm.trange(num_steps, desc="Least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start: start+batch_size]
            batch_ys = ys[start: start+batch_size]
            gradient = vector_mean(
                [sqerror_gradient(x, y, guess) for x,y in zip(batch_xs, batch_ys)]
            )
            guess = gradient_step(guess, gradient, -learning_rate)
    return guess


def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, beta) ** 2 for x,y in zip(xs,ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)


def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
    x_sample, y_sample = zip(*pairs)
    beta = least_squares_fit(
        xs=x_sample,
        ys=y_sample,
        learning_rate=0.001,
        num_steps=5000,
        batch_size=25
    )
    return beta
