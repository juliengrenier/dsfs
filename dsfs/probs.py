from typing import Tuple
import math
import random
from collections import Counter
import matplotlib.pyplot as plt

def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x < 1 else 0

def uniform_cdf(x: float) -> float:
    if x < 0 : return 0
    if x < 1 : return x
    return 1

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mu)**2 / 2 / sigma **2) / (math.sqrt(2 * math.pi) * sigma))


def normal_cdf(x: float, mu:float = 0, sigma: float =1) -> float:
    return (1 + math.erf((x-mu)/ math.sqrt(2) / sigma)) /2


def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1, tolerance: float = 0.00001) -> float:
    if mu != 0 or sigma != 1: return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0
    hi_z = 10
    while (hi_z - low_z) > tolerance:
        mid_z = (low_z + hi_z) /2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
    return mid_z


def bernouilli_trial(p: float) -> int:
    return 1 if random.random() < p else 0


def binomial(n: int, p: float) -> int:
    return sum(bernouilli_trial(p)for _ in range(n))


def binomial_histogram(p: float, n: int, num_points: int) -> None:
    data = [binomial(n, p) for _ in range(num_points)]
    histogram = Counter(data)
    plt.bar(
        [ x - 0.5 for x in histogram.keys()],
        [v/num_points for v in histogram.values()],
        0.8,
        color='0.75'
    )
    mu, sigma = normal_approximation_to_binomial(n, p)

    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title('Binomal Distribution vs Normal Approximation')
    plt.show()


def normal_approximation_to_binomial(n: int, p:float) -> Tuple[float, float]:
    mu = n * p
    sigma = math.sqrt(mu * (1-p))
    return mu, sigma



normal_probability_below = normal_cdf

def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    return 1 - normal_probability_below(lo, mu, sigma)


def normal_probability_between(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


def normal_probability_outside(lo, float, hi: float, mu: float = 0, sigma: float = 1 ) -> float:
    return 1 - normal_probability_between(lo, hi, mu, sigma)


def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    return inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    return inverse_normal_cdf(1-probability, mu, sigma)


def normal_two_sided_bounds(probability: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
    tail_probability = (1-probability) / 2
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound


def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)


upper_p_value = normal_probability_above

lower_p_value = normal_probability_below



def B(alpha: float, beta: float) -> float:
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:
        return 0
    return x ** (alpha -1) * (1 -x) ** (beta -1) / B(alpha, beta)

