import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time
import warnings
warnings.filterwarnings('ignore')
import sys
from tqdm import tqdm
from multiprocessing import Pool

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


def aaa(folder):
    logging.info('enter' + folder)
    with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)

    df = pd.read_csv(folder + 'train_data_idx.csv')
    df.drop('target', axis=1, inplace=True)
    df_val = df  # .loc[val, :].copy()

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

df_val = aaa('./0705_old_rate001/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
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


with open('map_user_order_num.pkl', 'rb') as f:
    map_user_order_num = pickle.load(f)

with open('map_reoder_rate.pkl', 'rb') as f:
    map_reoder_rate = pickle.load(f)

with open('map_order_cv_final.pkl', 'rb') as f:
    map_order_cv_final = pickle.load(f)

df_a = pd.read_csv('../input/df_train.csv', usecols=['product_id', 'reordered'], dtype=int)
# set_product = set(df_a[df_a['reordered'] == 1]['product_id'].unique().tolist())
set_product = set(df_a['product_id'].unique().tolist())


def uuu(args):
    order_id, vals = args
    items = [int(product_id) for product_id, _, _, _, _ in vals]
    # preds = np.array([pred_val if items[i] in set_product else 0 for i, (_, pred_val, _, _, _) in enumerate(vals)])
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    user_id = vals[0][4]

    none_prob = (1 - preds).prod()  # min(1 - map_reoder_rate[user_id], (1 - preds).prod())
    preds = np.r_[preds, [none_prob]]
    items.append('None')

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]

    items = [items[i] for i in idx]  # items[idx]
    none_idx = idx[-1]
    scores = []
    num_y_true = preds.sum()
    tp = 0
    ans = map_result.get(order_id, ['None'])

    for i in range(len(preds)):
        tp += preds[i]
        precision = tp / (i + 1)
        recall = tp / num_y_true
        f1 = (2 * precision * recall) / (precision + recall)
        scores.append((f1, i))

    f1, idx = max(scores, key=lambda x: x[0])

    data = [f1, idx, len(items), preds[idx], preds[min(idx + 1, len(items) - 1)], preds.sum(), preds.mean(), preds.min(), preds.max(),
            mean, std, map_user_order_num[user_id], map_reoder_rate[user_id]]
    data = [order_id, user_id] + data
    str_data = ",".join(map(str, data))
    with open('final_data/%s.csv' % order_id, 'w') as f:
        f.write(str_data + '\n')

    # idx = int(np.around(np.mean(idxs)))
    score = items[:idx + 1]

    ans = map_result.get(order_id, ['None'])
    sc = multilabel_fscore(ans, score)
    # print(ans, score, f1, sc)
    return sc


def sss2(map_pred):
    res = []
    p = Pool(7)
    aaaa = sorted(map_pred.items(), key=lambda x: x[0])
    res = p.map(uuu, tqdm(aaaa))
    # res = list(map(uuu, tqdm(aaaa)))
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
