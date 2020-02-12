from typing import List, Dict
from collections import Counter

import math
import matplotlib.pyplot as plt

from dsfs.matrix import Matrix, make_matrix
from dsfs.stats import correlation


def bucketize(point: float, bucket_size: float) -> float:
    return bucket_size * math.floor(point/bucket_size)


def make_histogram(points: List[float], bucket_size: float ) -> Dict[float, int]:
    return Counter(bucketize(point, bucket_size=bucket_size) for point in points)


def plot_histogram(points: List[float], bucket_size:float, title: str = ''):
    histogram = make_histogram(points, bucket_size=bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()


def correlation_matrix(data: Matrix) -> Matrix:
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])

    return make_matrix(len(data), len(data), correlation_ij)


def plot_correlation_matrix(corr_data: Matrix):
    num_vectors = len(corr_data)
    fig, ax = plt.subplots(num_vectors, num_vectors)
    for i in range(num_vectors):
        for j in range(num_vectors):
            if i != j:
                ax[i][j].scatter(corr_data[j], corr_data[i])
            else:
                ax[i][j].annotate(f'series {i}', (0.5,0.5), xycoords='axes fraction', ha='center', va='center')
            if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
            if j < num_vectors - 1: ax[i][j].yaxis.set_visible(False)
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())
