import numpy as np
cimport numpy as np
from sklearn.metrics import f1_score

def f1(np.ndarray[long, ndim=1] _label, np.ndarray[double, ndim=1] _preds):

    cdef long n = _preds.shape[0] + 1
    cdef np.ndarray[long, ndim=1] label = np.zeros(n, dtype=np.int)
    cdef np.ndarray[double, ndim=1] preds = np.zeros(n, dtype=np.float)

    label[:n - 1] = _label
    preds[:n - 1] = _preds

    label[n-1] = (1 - _preds).prod()
    if _label.sum() == 0:
        preds[n-1] = 1
        
    cdef np.ndarray[long, ndim=1] idx = np.argsort(preds)[::-1]
    label = label[idx]
    preds = preds[idx]
    cdef np.ndarray[double, ndim=1] scores = np.zeros(n)

    cdef double tp = 0.
    cdef double n_l = preds.sum()
    cdef long i
    cdef double precision, recall, f1
    
    for i in range(n):
        tp += preds[i]
        if tp > 0:
            precision = tp / (i + 1)
            recall = tp / n_l
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        scores[i] = f1
    i = np.argsort(scores)[n-1]
    tp = label[:i + 1].sum()
    precision = tp / (i + 1)
    recall = tp / label.sum()
    if tp > 0:
       f1 = (2 * precision * recall) / (precision + recall)
    else:
       f1 = 0
    return f1

# Should include None before
def f1_idx(np.ndarray[double, ndim=1] preds):

    cdef long n = preds.shape[0]
        
    cdef np.ndarray[long, ndim=1] idx = np.argsort(preds)[::-1]
    preds = preds[idx]
    cdef np.ndarray[double, ndim=1] scores = np.zeros(n)

    cdef double tp = 0.
    cdef double n_l = preds.sum()
    cdef long i
    cdef double precision, recall, f1
    
    for i in range(n):
        tp += preds[i]
        if tp > 0:
            precision = tp / (i + 1)
            recall = tp / n_l
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        scores[i] = f1
    i = np.argsort(scores)[n-1]

    return i + 1
