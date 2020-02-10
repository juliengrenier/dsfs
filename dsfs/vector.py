import math
from typing import List
Vector = List[float]

def add(v1:Vector, v2:Vector) -> Vector:
    assert len(v1) == len(v2), "vectors must be the same length"
    return [v1_i + v2_i for v1_i, v2_i in zip(v1, v2)]


def substract(v1:Vector, v2:Vector) -> Vector:
    assert len(v1) == len(v2), "vectors must be the same length"
    return [v1_i - v2_i for v1_i, v2_i in zip(v1, v2)]


def multiply(v1:Vector, v2:Vector) -> Vector:
    assert len(v1) == len(v2), "vectors must be the same length"
    return [v1_i * v2_i for v1_i, v2_i in zip(v1, v2)]


def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, "cannot be empty"
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "All vectors must be the same length"

    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)
    ]


def scalar_multiply(c:float, v:Vector) -> Vector:
    return [c * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot_product(v1: Vector, v2: Vector) -> float:
    return sum(multiply(v1, v2))


def sum_of_squares(v: Vector) -> float:
    return dot_product(v, v)


def magnitude(v: Vector) -> float:
    math.sqrt(sum_of_squares(v))


def squared_distance(v1: Vector, v2: Vector) -> float:
    return sum_of_squares(substract(v1, v2))


def distance(v1: Vector, v2: Vector) -> float:
    return math.sqrt(squared_distance(v1, v2))

