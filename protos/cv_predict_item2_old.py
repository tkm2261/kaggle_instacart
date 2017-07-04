import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time

import warnings
warnings.filterwarnings('ignore')
import sys

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
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred)
    recall = tp / len(y_true)

    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)

with open('item_info.pkl', 'rb') as f:
    map_pred, map_result = pickle.load(f)


np.random.seed(0)


def get_y_true(vals):
    y_true = [product_id
              for product_id, pred_val, mean, std, _ in vals if pred_val > np.random.uniform()]
    if len(y_true) == 0:
        y_true = ['None']
    return y_true

from tqdm import tqdm


from multiprocessing import Pool


def uuu(args):
    order_id, vals = args

    #sum_pred = sum(pred_val for _, pred_val, _, _,_ in vals)
    preds = np.array([pred_val for _, pred_val, _, _,_ in vals])
    vals += [('None', (1 - preds).prod(), 0, 0, None)]
    vals = sorted(vals, key=lambda x: x[1], reverse=True)
    items = [product_id for product_id, _, _, _,_ in vals]
    scenario = [get_y_true(vals) for _ in range(10000)]

    scores = []
    for i in range(len(vals)):
        pred = items[:i + 1]
        f1 = np.mean([multilabel_fscore(sc, pred) for sc in scenario])
        scores.append((f1, pred))
    f1, score = max(scores, key=lambda x: x[0])
    ans = map_result.get(order_id, ['None'])
    sc = multilabel_fscore(ans, score)
    #print(ans, score, f1, sc)
    
    return sc


def sss2(map_pred):
    res = []
    p = Pool()
    aaaa = sorted(map_pred.items(), key=lambda x: x[0])

    #res = p.map(uuu, aaaa)
    res = list(map(uuu, tqdm(aaaa)))
    p.close()
    p.join()
    return np.array(res)  # np.mean(res)

sc = sss2(map_pred)
score = np.mean(sc)
print(score, np.mean(sc))
# pd.DataFrame({174: rrr[0], 194: rrr[1]}).to_csv('tmp.csv', index=False)
