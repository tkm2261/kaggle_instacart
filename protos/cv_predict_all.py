import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time

import warnings
warnings.filterwarnings('ignore')


def aaa(folder):
    t = time.time()
    print('start', time.time() - t)
    with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)

    print('start1', time.time() - t)
    df = pd.read_csv(folder + 'train_data_idx.csv')
    df.drop('target', axis=1, inplace=True)
    print('start2', time.time() - t)
    df_val = df
    df_val['pred'] = pred

    print('start3', time.time() - t)

    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id'])

    df['target'] = 1

    print('start4', time.time() - t)
    df = pd.merge(df, df_val, how='outer', on=['order_id', 'user_id', 'product_id'])
    print('start5', time.time() - t)

    df['label'] = df['target'] == 1
    return df

df = aaa('./')
print('size', df.shape)
# df = aaa('./only_rebuy/')
# df = df.append(df2)
# df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()
df = df.reset_index(drop=True)
idxes = df.groupby('order_id').apply(lambda x: x.index.values).values


def exp_f1(label, pred):

    tp = sum(pred[i] for i in range(len(pred)) if label[i])
    fp = sum(pred[i] for i in range(len(pred)) if not label[i])
    fn = sum(1 - pred[i] for i in range(len(pred)) if label[i])

    f1 = 2 * tp / (2 * tp + fn + fp)
    return f1


def bbb(data):
    tmp = data.astype(int)
    sc = exp_f1(tmp[:, 0], tmp[:, 1])  # f1_score(tmp[:, 0], tmp[:, 1])
    return sc

from multiprocessing import Pool

for thresh in range(170, 200):
    thresh /= 1000
    df['pred_label'] = df['pred'] > thresh
    df['tp'] = (df['label'] == 1) & (df['pred_label'] == 1)
    df['tn'] = (df['label'] == 1) & (df['pred_label'] == 0)
    # scores = df.groupby('order_id', sort=False).apply(lambda tmp: f1_score(tmp['label'], tmp['pred_label']))
    scores = []
    aaa = df.values
    p = Pool()
    scores = list(p.map(bbb, [aaa[idx, 5:] for idx in idxes]))
    p.close()
    p.join()

    print(thresh, f1_score(df.label.values, df.pred_label.values), df.tp.sum(), df.tn.sum(), np.mean(scores))
