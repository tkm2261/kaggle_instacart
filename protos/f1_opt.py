# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
from tqdm import tqdm
from utils import f1_opt, f1
import logging
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s'

logging.basicConfig(format=log_fmt, level=logging.DEBUG)


def multilabel_fscore(y_true, y_pred):
    y_true, y_pred = set(y_true), set(y_pred)
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred)
    recall = tp / len(y_true)

    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


'''
def aaa(folder):
    logging.info('enter' + folder)
    with open(folder + 'train_cv_pred_0.pkl', 'rb') as f:
        pred = pickle.load(f)
    # with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
    #    pred = pickle.load(f)

    df = pd.read_csv('train_data_idx.csv')
    df.drop('target', axis=1, inplace=True)

    with open('user_split.pkl', 'rb') as f:
        cv = pickle.load(f)

    user_ids = df['user_id']
    for train, test in cv[: 1]:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)

    df_val = df.loc[val, :].copy()
    print(df_val.shape, pred.shape)
    df_val['pred'] = pred
    logging.info('exit')
    return df_val


def make_result():
    logging.info('enter')
    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id', 'reordered'])
    # df = df[df['user_id'].isin(test)].copy()

    map_result = {}
    for row in df[['order_id', 'user_id', 'product_id', 'reordered']].values:
        order_id, user_id, product_id, reordered = row
        if reordered == 0:
            continue
        order_id = int(order_id)
        if order_id not in map_result:
            map_result[order_id] = []

        map_result[order_id].append(int(product_id))
    logging.info('exit')
    return map_result


logging.info('user mean')
map_user_mean = pd.read_csv('../input/user_mean_order.csv', index_col='user_id').to_dict('index')
map_result = make_result()
df_val = aaa('./result_0803_1800/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val1 = aaa('./result_0803/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
df_val1 = aaa('./result_0728_18000/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)

df_val['pred'] = np.mean(np.vstack([df_val.pred.values, df_val1.pred.values]), axis=0)

df_val = df_val.sort_values(['order_id', 'pred'], ascending=False)
df_val = df_val[['order_id', 'user_id', 'product_id', 'pred']].values

map_pred = {}
n = df_val.shape[0]
for i in tqdm(range(n)):
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
'''

with open('item_info.pkl', 'rb') as f:
    map_pred, map_result = pickle.load(f)


def add_none(num, sum_pred, safe):
    score = 2. / (2 + num)
    if sum_pred * safe > score:
        return False
    else:
        return True


def _uuu(args):
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
    ans = map_result.get(order_id, ['None'])
    sc = multilabel_fscore(ans, score)
    # print(ans, score, f1, sc)

    return sc


def uuu(args):
    order_id, vals = args
    items = [int(product_id) for product_id, _, _, _, _ in vals]
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    ans = set(map_result.get(order_id, ['None']))

    label = np.array([i in ans for i in items], dtype=np.int)

    sc = f1_opt(label, preds)

    return sc


def sss2(map_pred):
    res = []
    p = Pool()
    aaaa = sorted(map_pred.items(), key=lambda x: x[0])
    res = p.map(uuu, tqdm(aaaa))
    #res = list(map(uuu, tqdm(aaaa)))
    p.close()
    p.join()
    return np.array(res)  # np.mean(res)


logging.info('start')
# idx = pd.read_csv('tmp_use.csv', header=None)[0].values
sc = sss2(map_pred)
score = np.mean(sc)
print(score, np.mean(sc))
# pd.DataFrame({174: rrr[0], 194: rrr[1]}).to_csv('tmp.csv', index=False)
logging.info('end')
