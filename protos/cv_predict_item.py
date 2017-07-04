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

    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id', 'reordered'])
    # df = df[df['user_id'].isin(test)].copy()

    map_result = {}
    for row in df[['order_id', 'user_id', 'product_id', 'reordered']].values:
        order_id, user_id, product_id, label = row
        order_id = int(order_id)
        if label == 0:
            continue
 
        if order_id not in map_result:
            map_result[order_id] = []
        
        map_result[order_id].append(int(product_id))
    for order_id, val in map_result.items():
        map_result[order_id] = val if len(val) != 0 else ["None"]
        
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
    map_pred[order_id].append((product_id, pred, mean, std, user_id))


with open('item_info.pkl', 'wb') as f:
    pickle.dump((map_pred, map_result), f, -1)

with open('item_info.pkl', 'rb') as f:
    map_pred, map_result = pickle.load(f)


def add_none(num, sum_pred, safe):
    score = 2. / (2 + num)
    if sum_pred * safe > score:
        return []
    else:
        return ['None']

with open('user_split.pkl', 'rb') as f:
    cv = pickle.load(f)

list_cv = []
map_user_cv = {}
for i, (train, test) in enumerate(cv):
    for u in test:
        map_user_cv[u] = i
    
def sss(map_pred, thresh, safe):
    res = [[] for _ in range(len(cv))]
    for order_id in sorted(map_pred.keys()):
        vals = map_pred[order_id]
        pred = []
        sum_pred = 0.
        for product_id, pred_val, mean, std, user_id in vals:
            if len(pred) > mean:
                break

            if pred_val > thresh:
                pred.append(product_id)
                sum_pred += pred_val

        if sum_pred > 0:
            sum_pred = 2 * sum_pred / (len(pred) + sum_pred)
        pred += add_none(len(pred), sum_pred, safe)
        #if sum_pred == 0:
        #    pred.append('None')
        ans = map_result.get(order_id, ['None'])

        res[map_user_cv[user_id]].append(multilabel_fscore(ans, pred))
    return np.array([np.mean(a) for a in res])  # np.mean(res)


rrr = []
idx = pd.read_csv('tmp_use.csv', header=None)[0].values
for i in [10]:
    safe = i / 10
    max_score = -1
    max_thresh = -1
    for thresh in range(150, 200):
        thresh /= 1000
        sc = sss(map_pred, thresh, safe)
        rrr.append(sc)
        score = np.mean(sc)
        print(thresh, score, sc)
        if max_score < score:
            max_score = score
            max_thresh = thresh
    print(safe, max_thresh, max_score, np.mean(sc))

#pd.DataFrame({174: rrr[0], 194: rrr[1]}).to_csv('tmp.csv', index=False)
