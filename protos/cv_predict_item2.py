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

#df_val = aaa('./result_0731_markov_cont8000/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val = aaa('./result_0730_markov/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
df_val = aaa('./result_0728_18000/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val1 = aaa('./0705_new/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val.pred += df_val1.pred.values
#df_val = aaa('./0710_stack2/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val = aaa('./0705_old_rate001/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)

#df_val = aaa('./0715_2nd_order/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val1 = aaa('./0714_10000loop/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val = aaa('./0716_3rd_order_stack/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val = aaa('./').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)

# df_val = aaa('./0706_tuned/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val1 = aaa('./0708_gpu_ids/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val2 = aaa('./0708_ids/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val.pred += df_val1.pred.values
# df_val = aaa('./0707_stack/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val.pred += df_val1.pred.values
# df_val1 = aaa('./0705_new_rate001/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)

#df_val['pred'] = np.mean(np.vstack([df_val.pred.values, df_val1.pred.values]), axis=0)
# df_val.pred = np.max(np.vstack([df_val.pred.values, df_val2.pred.values]), axis=0)


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


np.random.seed(0)
NUM = 10000
IS_COV = False
with open('map_user_order_num.pkl', 'rb') as f:
    map_user_order_num = pickle.load(f)

with open('map_reoder_rate.pkl', 'rb') as f:
    map_reoder_rate = pickle.load(f)


df_a = pd.read_csv('../input/df_train.csv', usecols=['product_id', 'reordered'], dtype=int)
# set_product = set(df_a[df_a['reordered'] == 1]['product_id'].unique().tolist())
set_product = set(df_a['product_id'].unique().tolist())


def _uuu(args):
    order_id, vals = args
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    items = [int(product_id) for product_id, _, _, _, _ in vals]
    user_id = vals[0][4]
    mean = vals[0][2]
    std = vals[0][3]

    none_prob = (1 - preds).prod()  # min(1 - map_reoder_rate[user_id], (1 - preds).prod())
    preds = np.r_[preds, [none_prob]]
    items.append('None')

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]

    items = [items[i] for i in idx]  # items[idx]

    scores = []
    tp = 0
    ans = map_result.get(order_id, ['None'])
    for i in range(len(preds)):
        f1 = multilabel_fscore(ans, items[:i + 1])
        scores.append((f1, i))
    _, best_idx = max(scores, key=lambda x: x[0])

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
    # idx = int(np.around(map_order_cv_final[order_id]))
    idx = max(0, idx)
    idx = min(len(items) - 1, idx)
    score = items[:idx + 1]
    sc = multilabel_fscore(ans, score)

    if preds.max() > 1:
        print('AAAAA', order_id)

    label = best_idx
    data = [f1, idx, len(items), preds[idx], preds[min(idx + 1, len(items) - 1)], preds.sum(), preds.mean(), preds.min(), preds.max(),
            mean, std, map_user_order_num[user_id], map_reoder_rate[user_id]]
    data = [order_id, user_id, label] + data
    str_data = ",".join(map(str, data))
    with open('final_data/%s.csv' % order_id, 'w') as f:
        f.write(str_data + '\n')

    sc = multilabel_fscore(ans, score)
    # print(ans, score, f1, sc)
    return sc

#map_user_mean_r = pd.read_csv('../input/user_mean_order_reordered.csv', index_col='user_id').to_dict('index')

def uuu(args):
    order_id, vals = args
    items = [int(product_id) for product_id, _, _, _, _ in vals]
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    user_id = vals[0][4]
    mean = vals[0][2]

    none_prob = (1 - preds).prod()  # min(1 - map_reoder_rate[user_id], (1 - preds).prod())
    preds = np.r_[preds, [none_prob]]

    items.append('None')

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]
    items = [items[i] for i in idx]  # items[idx]
    none_idx = idx[-1]
    idxs = []
    _preds = preds.copy()
    for _ in range(1):
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
        idxs.append(idx)

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
