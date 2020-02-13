from dsfs.vector import Vector, direction, substract, dot_product, project
from dsfs.matrix import Matrix, directional_variance, directional_variance_gradient
from dsfs.gradients import gradient_step

import tqdm


def first_principal_component(matrix: Matrix, n: int = 100, step_size: float = 0.1) -> Vector:
    guess = [1.0] * len(matrix[0])
    display(guess)

    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(matrix, guess)
            gradient = directional_variance_gradient(matrix, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description("dv: {dv:.3f}")
    return direction(guess)


def remove_projection_from_vector(v1: Vector, v2: Vector) -> Vector:
    return substract(v1, project(v1, v2))


def remove_projection(matrix: Matrix, w: Vector) -> Matrix:
    return [remove_projection_from_vector(v, w) for v in matrix]


def pca(matrix: Matrix, num_components: int) -> Matrix:
    components: Matrix = []
    for _ in range(num_components):
        component = first_principal_component(matrix)
        components.append(component)
        data = remove_projection(matrix, component)
    return components


def transform_vector(v: Vector, components: Matrix) -> Vector:
    return [dot_product(v, w) for w in components]


def transform(matrix: Matrix, components: Matrix) -> Matrix:
    return [transform_vector(v, components) for v in matrix]
