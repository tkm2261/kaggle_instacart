import numpy as np
cimport numpy as np

def f1(np.ndarray[long, ndim=1] label, np.ndarray[double, ndim=1] preds):

    cdef long n = preds.shape[0]
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
       precision = tp / (i + 1)
       recall = tp / n_l
       f1 = (2 * precision * recall) / (precision + recall)
    else:
       f1 = 0

    return f1

