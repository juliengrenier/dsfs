import random
from typing import Callable, List, TypeVar, Iterator
from dsfs.vector import Vector, scalar_multiply, distance, add, vector_mean

def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x+h) - f(x))/ h


def partial_difference_quotient(
        f: Callable[[Vector], float],
        v: Vector,
        i: int,
        h, float) -> float:
    w = [w_j + (h if j == i else 0) for j,v_j in enumerate(v)]
    return (f(w) - f(v))/h


def estimate_gradient(
        f: Callable[[Vector], float],
        v: Vector,
        h: float = 0.0001) -> List[float]:

    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad


T = TypeVar('T')
def minibatches(dataset: Iterator[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    dataset = list(dataset)
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle: random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start: end]



def linear_gradient_descent(
        xs: List[float],
        ys: List[float],
        num_epochs: int,
        learning_rate: float,
        initial_weights: List[float]) -> Iterator[List[float]]:
    theta = initial_weights
    for epoch in range(num_epochs):
        grad = vector_mean([linear_gradient(x, y, theta) for x,y in zip(xs, ys)])
        theta = gradient_step(theta, gradient=grad, step_size=-learning_rate)
        yield epoch, theta


def minibatch_linear_gradient_descent(
        xs: List[float],
        ys: List[float],
        num_epochs: int,
        learning_rate: float,
        initial_weights: List[float],
        batch_size: int = 20,
        shuffle: bool = False) -> Iterator[List[float]]:
    theta = initial_weights
    for epoch in range(num_epochs):
        for batch in minibatches(zip(xs, ys), batch_size=batch_size, shuffle=shuffle):
            grad = vector_mean([linear_gradient(x, y, theta) for x,y in batch])
            theta = gradient_step(theta, gradient=grad, step_size=-learning_rate)
        yield epoch, theta

def stochastic_linear_gradient_descent(
        xs: List[float],
        ys: List[float],
        num_epochs: int,
        learning_rate: float,
        initial_weights: List[float]) -> Iterator[List[float]]:
    theta = initial_weights
    for epoch in range(num_epochs):
        for x, y in zip(xs, ys):
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, gradient=grad, step_size=-learning_rate)
        yield epoch, theta
