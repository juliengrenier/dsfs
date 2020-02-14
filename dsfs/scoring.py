
# tp -> true positive
# fp -> false positive
# fn -> false negative
# tn -> true negative

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    "The proportion of correct predictions"
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct/total


def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    "Precision measures the accuracy of the positive predictions"
    return tp / (tp + fp)


def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    "Recall measures the proportion of the positives identified"
    return tp / (tp + fn)


def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tp)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)
