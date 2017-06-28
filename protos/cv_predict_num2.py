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
    # with open(folder + 'train_cv_pred.pkl', 'rb') as f:
    #    pred = pickle.load(f)
    with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)

    print('start1', time.time() - t)
    df = pd.read_csv(folder + 'train_data_idx.csv')

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
    #df = df[df['user_id'].isin(test)].copy()
    df['target'] = 1

    print('start4', time.time() - t)
    df = pd.merge(df, df_val, how='outer', on=['order_id', 'user_id', 'product_id'])
    print('start5', time.time() - t)

    df['label'] = (df['target'] == 1).astype(np.bool)
    df['pred'] = df['pred'].fillna(0)
    return df

df = aaa('./')


print('size', df.shape)

map_user_mean = pd.read_csv('../input/user_mean_order.csv', index_col='user_id').to_dict('index')

from multiprocessing import Pool


def bbb(args):
    data, thresh = args

    y_true = data[:, 0].astype(np.bool)
    pred = data[:, 1]
    idx = np.argsort(pred)[::-1]
    y_true = y_true[idx]
    pred = pred[idx]

    user_id = data[0, 2]
    tmp = map_user_mean[user_id]
    mean = tmp['mean']
    std = tmp['std']

    mean = int(mean + 2 * std)

    y_pred = (pred > thresh).astype(np.bool)
    y_pred[mean:] = False

    sc = f1_score(y_true, y_pred)
    return sc, y_true, y_pred


max_score = 0
max_thresh = None
df = df.reset_index(drop=True)
idxes = df.groupby('order_id').apply(lambda x: x.index.values).values


for thresh in [194]:  # range(170, 210):
    thresh /= 1000
    df['pred_label'] = df['pred'] > thresh
    # scores = df.groupby('order_id', sort=False).apply(lambda tmp: f1_score(tmp['label'], tmp['pred_label']))
    scores = []

    aaa = df[['label', 'pred', 'user_id']].values

    p = Pool()
    tmp = list(p.map(bbb, [(aaa[ii], thresh) for ii in idxes]))
    p.close()
    p.join()
    scores = [t[0] for t in tmp]
    y_true = np.concatenate([t[1] for t in tmp])
    y_pred = np.concatenate([t[2] for t in tmp])

    print(thresh,
          f1_score(df.label.values, df.pred_label.values),
          f1_score(y_true, y_pred),
          np.mean(scores))
