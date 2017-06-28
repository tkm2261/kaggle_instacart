import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb

import time

import warnings
warnings.filterwarnings('ignore')


def aaa(folder):
    t = time.time()
    print('start', time.time() - t)
    with open(folder + 'train_cv_pred.pkl', 'rb') as f:
        pred = pickle.load(f)

    print('start1', time.time() - t)
    df = pd.read_csv(folder + 'train_data_idx.csv')

    print('start2', time.time() - t)
    with open(folder + 'user_split.pkl', 'rb') as f:
        cv = pickle.load(f)

    list_cv = []
    user_ids = df['user_id']

    for train, test in cv[:1]:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)
        list_cv.append((trn, val))

    df_val = df.loc[val, :].copy()
    df_val['pred'] = pred

    df_order = df_val.groupby('order_id')['pred'].agg(
        {'o_avg': np.mean, 'o_min': np.min, 'o_max': np.max, 'o_cnt': len}).reset_index()
    df_user = df_val.groupby('user_id')['pred'].agg(
        {'u_avg': np.mean, 'u_min': np.min, 'u_max': np.max, 'u_cnt': len}).reset_index()
    df_item = df_val.groupby('product_id')['pred'].agg(
        {'p_avg': np.mean, 'p_min': np.min, 'p_max': np.max, 'p_cnt': len}).reset_index()

    df_val = pd.merge(df_val, df_order, how='left', on='order_id')
    df_val = pd.merge(df_val, df_user, how='left', on='user_id')
    df_val = pd.merge(df_val, df_item, how='left', on='product_id').sort_values('order_id')

    df_val['pred2'] = df_val.pred
    df_val.drop('target', inplace=True, axis=1)
    print('start3', time.time() - t)

    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id'])
    df = df[df['user_id'].isin(test)].copy()
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


def bbb(data):
    tmp = data.astype(int)
    sc = f1_score(tmp[:, 0], tmp[:, 1])
    return sc
df_order = df.groupby('order_id')['pred'].agg({'cnt_order': len}).reset_index()
_df = pd.merge(df, df_order, how='left', on='order_id')

from multiprocessing import Pool

# for thresh in range(100, 200):
ra = list(range(0, 100, 20)) + [100, 200, 1000]
#ra = [0, 1000]
list_thresh = []
for i in range(len(ra) - 1):
    df = _df[(_df.cnt_order >= ra[i]) & (_df.cnt_order < ra[i + 1])].copy()
    max_score = 0
    max_thresh = None
    for thresh in range(100, 200):
        thresh /= 1000

        df['pred_label'] = df['pred'] > thresh
        df['pred_label2'] = df['pred2'] > thresh
        sc = f1_score(df.label.values, df.pred_label.values)
        if max_score < sc:
            max_score = sc
            max_thresh = thresh
    scores = []

    list_thresh.append(max_thresh)
    print(ra[i], ra[i + 1], max_thresh, max_score)

with open('num_thresh.pkl', 'wb') as f:
    pickle.dump((ra, list_thresh), f, -1)


def predict(val, num):
    if np.isnan(val):
        return False
    for i in range(len(ra) - 1):
        if num >= ra[i] and num < ra[i + 1]:
            return val > list_thresh[i]
    raise

_df['pred_label'] = _df.apply(lambda row: predict(row.pred, row.cnt_order), axis=1)
sc = f1_score(_df.label.values, _df.pred_label.values)

print(sc)
