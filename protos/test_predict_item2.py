import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

import time
from tqdm import tqdm
import logging
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s'

logging.basicConfig(format=log_fmt, level=logging.DEBUG)


def aaa(folder):
    t = time.time()
    print('start', folder)
    df = pd.read_csv('test_data_idx.csv').sort_values(['order_id', 'user_id', 'product_id'])

    with open(folder + 'test_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    return df


df = aaa('result_0803/')
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

np.random.seed(0)
NUM = 10000


from multiprocessing import Pool


def uuu(args):
    order_id, vals = args

    items = [int(product_id) for product_id, _, _, _, _ in vals]
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    user_id = vals[0][4]

    none_prob = (1 - preds).prod()  # min(1 - map_reoder_rate[user_id], (1 - preds).prod())
    preds = np.r_[preds, [none_prob]]
    items.append('None')

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]

    items = [items[i] for i in idx]  # items[idx]
    none_idx = idx[-1]
    idxs = []
    scores = []
    num_y_true = preds.sum()
    tp = 0

    for i in range(len(preds)):
        tp += preds[i]
        precision = tp / (i + 1)
        recall = tp / num_y_true
        f1 = (2 * precision * recall) / (precision + recall)
        scores.append((f1, i))

    f1, idx = max(scores, key=lambda x: x[0])
    score = items[:idx + 1]

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
