# -*- coding: utf-8 -*-
import time
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
from tqdm import tqdm

import logging
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s'

logging.basicConfig(format=log_fmt, level=logging.DEBUG)

'''
This kernel implements the O(n2) F1-Score expectation maximization algorithm presented in
"Ye, N., Chai, K., Lee, W., and Chieu, H.  Optimizing F-measures: A Tale of Two Approaches. In ICML, 2012."

It solves argmax_(0 <= k <= n,[[None]]) E[F1(P,k,[[None]])]
with [[None]] being the indicator for predicting label "None"
given posteriors P = [p_1, p_2, ... , p_n], where p_1 > p_2 > ... > p_n
under label independence assumption by means of dynamic programming in O(n2).
'''


class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)


def multilabel_fscore(y_true, y_pred):
    y_true, y_pred = set(y_true), set(y_pred)
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred)
    recall = tp / len(y_true)

    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def aaa(folder):
    t = time.time()
    print('start', folder)
    df = pd.read_csv('test_data_idx.csv').sort_values(['order_id', 'user_id', 'product_id'])

    with open(folder + 'test_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    return df


df = aaa('result_0803_1800/')
'''
df = aaa('./result_0728_18000/')
df1 = aaa('./result_0731_markov_cont8000/')
df2 = aaa('./result_0731_allfeat/')
df3 = aaa('./result_0730_markov/')

df['pred'] = np.sum(np.vstack([df.pred.values * 3,
                                df1.pred.values,
                                df2.pred.values,
                                df3.pred.values                             
                                    ]), axis=0) / 6
'''
# df2 = aaa('./only_rebuy/')
# df = df.append(df2)
# df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()

df = df.sort_values(['order_id', 'pred'], ascending=False)
df = df[['order_id', 'user_id', 'product_id', 'pred']].values

map_user_mean = pd.read_csv('../input/user_mean_order.csv', index_col='user_id').to_dict('index')

map_pred = {}
n = df.shape[0]
for i in tqdm(range(n)):
    order_id, user_id, product_id, pred = df[i]
    order_id, user_id, product_id = list(map(int, [order_id, user_id, product_id]))

    tmp = map_user_mean[user_id]
    mean = tmp['mean']
    std = tmp['std']
    if order_id not in map_pred:
        map_pred[order_id] = []
    map_pred[order_id].append((product_id, pred, mean, std, user_id))


def uuu(args):
    order_id, vals = args
    items = [int(product_id) for product_id, _, _, _, _ in vals]
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    pNone = (1 - preds).prod()  # min(1 - map_reoder_rate[user_id], (1 - preds).prod())

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]
    items = [items[i] for i in idx]  # items[idx]

    best_k, predNone, max_f1 = F1Optimizer.maximize_expectation(preds, pNone)
    score = items[:best_k]
    if predNone:
        score += ['None']
    return order_id, score


p = Pool()
#result = list(map(uuu, tqdm(map_pred.items())))
result = list(p.map(uuu, tqdm(map_pred.items())))
p.close()
p.join()

f = open('submit.csv', 'w')
f.write('order_id,products\n')
for key, val in sorted(result, key=lambda x: x[0]):
    val = " ".join(map(str, val))
    f.write('{},{}\n'.format(key, val))
f.close()
