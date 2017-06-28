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

    with open(folder + 'train_cv_pred.pkl', 'rb') as f:
        pred = pickle.load(f)
    """
    with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)
    """
    print('start1', time.time() - t)
    df = pd.read_csv(folder + 'train_data_idx.csv', dtype=int)
    df.drop('target', axis=1, inplace=True)
    print('start2', time.time() - t)

    with open(folder + 'user_split.pkl', 'rb') as f:
        cv = pickle.load(f)
    list_cv = []
    user_ids = df['user_id']

    for train, test in cv[:1]:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)
        list_cv.append((trn, val))
    #pred = pred[val]

    df_val = df.loc[val, :].copy()
    df_val['pred'] = pred
    print('start3', time.time() - t)

    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
    df = df[df['user_id'].isin(test)].copy()
    df['target'] = 1

    print('start4', time.time() - t)
    df = pd.merge(df, df_val, how='outer', on=['order_id', 'user_id', 'product_id'])
    print('start5', time.time() - t)
    df['target'] = df['target'].fillna(0)
    df['pred'] = df['pred'].fillna(0)

    df['label'] = df['target'] == 1
    return df

df = aaa('./')
print('size', df.shape)
# df = aaa('./only_rebuy/')
# df = df.append(df2)
# df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()
df = df.reset_index(drop=True)
idxes = df.groupby('order_id').apply(lambda x: x.index.values).values


def bbb(data):
    y_true = data[:, 0]
    y_pred = data[:, 1]

    sc = f1_score(y_true, y_pred)
    return sc

from multiprocessing import Pool

for thresh in range(170, 210):
    thresh /= 1000
    df['pred_label'] = df['pred'] > thresh
    # scores = df.groupby('order_id', sort=False).apply(lambda tmp: f1_score(tmp['label'], tmp['pred_label']))
    scores = []

    aaa = df[['label', 'pred_label']].values.astype(np.bool)
    p = Pool()
    scores = list(p.map(bbb, [aaa[idx] for idx in idxes]))
    p.close()
    p.join()
    print(thresh, f1_score(df.label.values, df.pred_label.values), np.mean(scores))
