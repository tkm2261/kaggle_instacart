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

'''
def aaa(folder):
    t = time.time()
    print('start', time.time() - t)
    # with open(folder + 'train_cv_pred.pkl', 'rb') as f:
    #    pred = pickle.load(f)
    with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)

    print('start1', time.time() - t)
    df = pd.read_csv(folder + 'train_data_idx.csv')
    df.drop('target', axis=1, inplace=True)
    print('start2', time.time() - t)
    """
    with open(folder + 'user_split.pkl', 'rb') as f:
        cv = pickle.load(f)

    list_cv = []
    user_ids = df['user_id']

    for train, test in cv[:1]:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)
        list_cv.append((trn, val))
    """
    df_val = df  # .loc[val, :].copy()

    df_val['pred'] = pred

    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id'])
    # df = df[df['user_id'].isin(test)].copy()

    map_result = {}
    for row in df[['order_id', 'user_id', 'product_id']].values:
        order_id, user_id, product_id = row
        order_id = int(order_id)
        if order_id not in map_result:
            map_result[order_id] = []

        map_result[order_id].append(int(product_id))

    test = df.order_id.unique()
    return df_val, test, map_result


map_user_mean = pd.read_csv('../input/user_mean_order.csv', index_col='user_id').to_dict('index')


df_val, test, map_result = aaa('./')
df_val = df_val.sort_values(['order_id', 'pred'], ascending=False)
df_val = df_val[['order_id', 'user_id', 'product_id', 'pred']].values

map_pred = {}
n = df_val.shape[0]
for i in range(n):
    order_id, user_id, product_id, pred = df_val[i]
    order_id, user_id, product_id = list(map(int, [order_id, user_id, product_id]))

    tmp = map_user_mean[user_id]
    mean = tmp['mean']
    std = tmp['std']
    if order_id not in map_pred:
        map_pred[order_id] = []
    map_pred[order_id].append((product_id, pred, mean, std))


with open('item_info.pkl', 'wb') as f:
    pickle.dump((map_pred, map_result), f, -1)
'''
with open('item_info.pkl', 'rb') as f:
    map_pred, map_result = pickle.load(f)


def add_none(num, sum_pred, safe):
    score = 2. / (2 + num)
    if sum_pred * safe > score:
        return []
    else:
        return ['None']


def exp_f1(label, pred):

    tp = sum(pred[i] for i in range(len(pred)) if label[i])
    fp = sum(pred[i] for i in range(len(pred)) if not label[i])
    fn = sum(1 - pred[i] for i in range(len(pred)) if label[i])

    f1 = 2 * tp / (2 * tp + fn + fp)
    return f1


def search(pred):
    if len(pred) == 0:
        return 0

    fp = np.sum(pred)
    fn = 0.
    tp = 0.

    f1 = 2 * tp / (2 * tp + fn + fp)

    for i, p in enumerate(pred):
        fp -= p
        fn += (1 - p)
        tp += p
        _f1 = 2 * tp / (2 * tp + fn + fp)

        if _f1 > f1:
            f1 = _f1
        else:
            break
    return i

np.random.seed(0)


def get_y_true(vals):
    y_true = [product_id
              for product_id, pred_val, mean, std in vals if pred_val > np.random.uniform()]
    if len(y_true) == 0:
        y_true = ['None']
    return y_true

from tqdm import tqdm


def sss(map_pred):
    res = []
    for order_id in tqdm(sorted(map_pred.keys())):
        vals = map_pred[order_id]
        sum_pred = sum(pred_val for _, pred_val, _, _ in vals)
        if sum_pred < 1:
            vals += [('None', 1 - sum_pred, 0, 0)]
            vals = sorted(vals, key=lambda x: x[1], reverse=True)
        items = [product_id for product_id, _, _, _ in vals]
        scenario = [get_y_true(vals) for _ in range(1000)]

        scores = []
        for i in range(len(vals)):
            pred = items[:i + 1]
            f1 = np.mean([multilabel_fscore(sc, pred) for sc in scenario])
            scores.append((f1, pred))
        f1, score = max(scores, key=lambda x: x[0])
        res.append(score)
    return np.array(res)  # np.mean(res)

from multiprocessing import Pool


def uuu(args):
    order_id, vals = args

    sum_pred = sum(pred_val for _, pred_val, _, _ in vals)
    if sum_pred < 1:
        vals += [('None', 1 - sum_pred, 0, 0)]
        vals = sorted(vals, key=lambda x: x[1], reverse=True)
    items = [product_id for product_id, _, _, _ in vals]
    scenario = [get_y_true(vals) for _ in range(1000)]

    scores = []
    for i in range(len(vals)):
        pred = items[:i + 1]
        f1 = np.mean([multilabel_fscore(sc, pred) for sc in scenario])
        scores.append((f1, pred))
    f1, score = max(scores, key=lambda x: x[0])
    ans = map_result.get(order_id, ['None'])
    sc = multilabel_fscore(ans, score)
    return sc


def sss2(map_pred):
    res = []
    p = Pool()
    aaaa = sorted(map_pred.items(), key=lambda x: x[0])
    res = p.map(uuu, aaaa)
    #res = list(map(uuu, tqdm(aaaa)))
    p.close()
    p.join()
    return np.array(res)  # np.mean(res)

idx = pd.read_csv('tmp_use.csv', header=None)[0].values
sc = sss2(map_pred)
score = np.mean(sc[idx])
print(score, np.mean(sc))
# pd.DataFrame({174: rrr[0], 194: rrr[1]}).to_csv('tmp.csv', index=False)
