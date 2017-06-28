from sklearn.metrics import f1_score
import numpy as np


def multilabel_fscore(y_true, y_pred):
    """
    ex1:
    y_true = [1, 2, 3]
    y_pred = [2, 3]
    return: 0.8

    ex2:
    y_true = ["None"]
    y_pred = [2, "None"]
    return: 0.666

    ex3:
    y_true = [4, 5, 6, 7]
    y_pred = [2, 4, 8, 9]
    return: 0.25

    """
    y_true, y_pred = set(y_true), set(y_pred)

    precision = sum([1 for i in y_pred if i in y_true]) / len(y_pred)

    recall = sum([1 for i in y_true if i in y_pred]) / len(y_true)

    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def multilabel_fscore2(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.bool_)
    y_pred = np.array(y_pred, dtype=np.bool_)
    tp = (y_true * y_pred).sum()
    precision = tp / y_pred.sum()
    recall = tp / y_true.sum()

    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


if __name__ == '__main__':
    #   [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = [0, 0, 0, 1, 1, 1, 1, 0, 0]
    b = [0, 1, 0, 1, 0, 0, 0, 1, 1]
    print(multilabel_fscore([4, 5, 6, 7], [2, 4, 8, 9]), multilabel_fscore2(a, b), f1_score(a, b))
