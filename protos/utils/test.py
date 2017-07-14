import pyximport; pyximport.install()
from utils import f1
from sklearn.metrics import f1_score
import numpy as np
def f2(_label, _preds):

    none_prob = (1 - _preds).prod()
    is_none = 1 if _label.sum() == 0 else 0
    _n = _label.shape[0]
    
    label = np.zeros(_n + 1, dtype=int)
    preds = np.zeros(_n + 1, dtype=float)

    label[:_n] = _label
    preds[:_n] = _preds

    label[_n] = is_none
    preds[_n] = none_prob

    idx = np.argsort(preds)[::-1]
    label = label[idx]
    preds = preds[idx]
    scores = []
    #
    tp = 0
    n_l = preds.sum()
    for i in range(len(preds)):
        #pl[i] = True
        #f1 = f1_score(label[:i + 1], pl[:i + 1])
        tp += preds[i]
        if tp > 0:
            precision = tp / (i + 1)
            recall = tp / n_l
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        scores.append((f1, i))
    f1, idx = max(scores, key=lambda x: x[0])
    
    tp = label[:idx + 1].sum()
    precision = tp / (idx + 1)
    recall = tp / label.sum()

    if tp > 0:
       f1 = (2 * precision * recall) / (precision + recall)
    else:
       f1 = 0
    print('aaa', f1_score(label, preds > preds[idx + 1]))
    print('sss', f1)    
    return f1

import time
import pandas as pd
df = pd.read_csv('test.csv')

none_prob = (1 - df.pred.values).prod()

preds = np.r_[df.pred.values, [none_prob]]
label = np.r_[df.target.values, [1 if df.target.sum() == 0 else 0]]

#print(f1_score(df.target.values, df.pred.values >= 0.2118))
#print(f1_score(label, preds >= 0.2118))
print(f2(df.target.values, df.pred.values))

"""
label = (np.random.random(10000) > 0.5).astype(int)
preds = np.random.random(10000)

t = time.time()
for _ in range(100):
    (f1(label, preds))
print(time.time() - t)

t = time.time()
for _ in range(100):
    (f2(label, preds))
print(time.time() - t)
"""
