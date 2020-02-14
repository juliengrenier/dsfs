from typing import List, NamedTuple
from collections import Counter

from dsfs.vector import Vector, distance

class LabeledPoint(NamedTuple):
    point: Vector
    label: str


def majority_vote(labels: List[str]) -> str:
    "Labels are orderd from nearest to farthest"
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])


def classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))

    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    return majority_vote(k_nearest_labels)
