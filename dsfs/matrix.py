from typing import List, Tuple, Callable

from dsfs.vector import Vector, vector_mean
from dsfs.stats import standard_deviation

Matrix = List[List[float]]


def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0] if A else 0)
    return num_rows, num_cols


def row(A: Matrix, i: int) -> Vector:
    return A[i]

def column(A: Matrix, j: int) -> Vector:
    return [A_i[j] for A_i in A]


def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j)
             for j in range(num_cols)]
           for i in range(num_rows)]


def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i,j: 1 if i == j else 0)

def zero_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i,j: 0)


def scale(matrix: Matrix) -> Tuple[float, float]:
    dim = len(matrix[0])
    means = vector_mean(matrix)
    stddevs = [standard_deviation([vector[i] for vector in matrix]) for i in range(dim)]
    return means, stddevs


def rescale(matrix: Matrix) -> Matrix:
    dim = len(matrix[0])
    means, stddevs = scale(matrix)
    rescaled = [v[:] for v in matrix]
    for v in rescaled:
        for i in range(dim):
            if stddevs[i] > 0:
                v[i] = (v[i] - means[i])/stddevs[i]
    return rescaled
