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

    model = LGBMClassifier(seed=0, max_depth=5, min_child_samples=50, n_estimators=100)
    data = df_val[['pred',
                   'o_avg', 'o_min', 'o_max', 'o_cnt',
                   'u_avg', 'u_min', 'u_max', 'u_cnt',
                   'p_avg', 'p_min', 'p_max', 'p_cnt', ]].values
    """
    model.fit(data, df_val['target'])
    with open('model2.pkl', 'wb') as f:
        pickle.dump(model, f, -1)
    """
    data = xgb.DMatrix(data, df_val['target'])
    df_val['order_id'] = df_val['order_id'].astype(int)

    group_val = df_val.groupby('user_id')['user_id'].count().sort_index().values

    data.set_group(group_val)
    model = xgb.train({'objective': 'rank:pairwise', 'max_depth': 5, 'min_child_weight': 30},
                      data,
                      num_boost_round=100
                      )  # .fit(data, df_val['target'])
    df_val['pred2'] = model.predict(data)  # _proba(data)[:, 1]
    #df_val['pred2'] = model.predict_proba(data)[:, 1]
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

from multiprocessing import Pool

# for thresh in range(100, 200):
for thresh in range(101):
    thresh /= 100
    df['pred_label'] = df['pred'] > thresh
    df['pred_label2'] = df['pred2'] > thresh

    df['aaa'] = (df['label'] == 1) & (df['pred_label'] == 1)
    # scores = df.groupby('order_id', sort=False).apply(lambda tmp: f1_score(tmp['label'], tmp['pred_label']))
    scores = []
    """
    aaa = df.values
    p = Pool()
    scores = list(p.map(bbb, [aaa[idx, 5:] for idx in idxes]))
    p.close()
    p.join()

    for idx in idxes:
        tmp = aaa[idx, 5:].astype(int)
        sc = f1_score(tmp[:, 0], tmp[:, 1])
        scores.append(sc)
    """
    print(thresh, f1_score(df.label.values, df.pred_label.values), f1_score(
        df.label.values, df.pred_label2.values), df.aaa.sum(), np.mean(scores))
