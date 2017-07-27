from cython.parallel cimport parallel, prange
from libc.stdlib cimport abort, malloc, free
cimport cython
import numpy as np
cimport numpy as np
from sklearn.metrics import f1_score


@cython.boundscheck(False)
@cython.wraparound(False)
def f1(np.ndarray[long, ndim=1] _label, np.ndarray[double, ndim=1] _preds):

    cdef long n = _preds.shape[0] + 1
    cdef np.ndarray[long, ndim= 1] label = np.zeros(n, dtype=np.int)
    cdef np.ndarray[double, ndim= 1] preds = np.zeros(n, dtype=np.float)

    label[:n - 1] = _label
    preds[:n - 1] = _preds

    preds[n - 1] = (1 - _preds).prod()
    if _label.sum() == 0:
        label[n - 1] = 1

    cdef np.ndarray[long, ndim= 1] idx = np.argsort(preds)[::-1]
    label = label[idx]
    preds = preds[idx]
    cdef np.ndarray[double, ndim= 1] scores = np.zeros(n)

    cdef double tp = 0.
    cdef double n_l = preds.sum()
    cdef long i
    cdef double precision, recall, f1, score

    score = -1
    for i in range(n):
        tp += preds[i]
        if tp > 0:
            precision = tp / (i + 1)
            recall = tp / n_l
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        #scores[i] = f1
        if f1 < score:
            i = i - 1
            break
        score = f1
    # i = i - 1  # np.argsort(scores)[n - 1]
    tp = label[:i + 1].sum()
    if tp > 0:
        precision = tp / (i + 1)
        recall = tp / label.sum()
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


@cython.boundscheck(False)
@cython.wraparound(False)
def f1_group(np.ndarray[long, ndim=1] label, np.ndarray[double, ndim=1] preds, np.ndarray[long, ndim=1] group):
    cdef int i, start, end, j, s
    cdef double score = 0.
    cdef long m = group.shape[0]
    cdef long n = preds.shape[0]
    start = 0
    for i in range(m):
        end = start + group[i]
        score += f1(label[start:end], preds[start:end])
        start = end
    return score / m

# Should include None before


@cython.boundscheck(False)
@cython.wraparound(False)
def f1_idx(np.ndarray[long, ndim=1] _label, np.ndarray[double, ndim=1] _preds):

    cdef long n = _preds.shape[0] + 1
    cdef np.ndarray[long, ndim= 1] label = np.zeros(n, dtype=np.int)
    cdef np.ndarray[double, ndim= 1] preds = np.zeros(n, dtype=np.float)

    label[:n - 1] = _label
    preds[:n - 1] = _preds

    preds[n - 1] = (1 - _preds).prod()
    if _label.sum() == 0:
        label[n - 1] = 1

    cdef np.ndarray[long, ndim= 1] idx = np.argsort(preds)[::-1]
    label = label[idx]
    preds = preds[idx]
    cdef np.ndarray[double, ndim= 1] scores = np.zeros(n)

    cdef double tp = 0.
    cdef double n_l = preds.sum()
    cdef long i
    cdef double precision, recall, f1, score

    score = -1
    for i in range(n):
        tp += preds[i]
        if tp > 0:
            precision = tp / (i + 1)
            recall = tp / n_l
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        #scores[i] = f1
        if f1 < score:
            i = i - 1
            break
        score = f1
    return preds[i]


@cython.boundscheck(False)
@cython.wraparound(False)
def f1_group_idx(np.ndarray[long, ndim=1] label, np.ndarray[double, ndim=1] preds, np.ndarray[long, ndim=1] group):
    cdef int i, start, end, j, s
    cdef double score = 0.
    cdef long m = group.shape[0]
    cdef long n = preds.shape[0]
    cdef np.ndarray[long, ndim= 1] res = np.zeros(n, dtype=np.int)

    start = 0
    for i in range(m):
        end = start + group[i]
        thresh = f1_idx(label[start:end], preds[start:end])
        res[start:end] = (preds[start:end] >= thresh) == label[start:end]
        # for j in range(start, end):
        #    res[j] = (preds[j] >= thresh) == label[j]
        start = end
    return res
