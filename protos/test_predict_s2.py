import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time


def aaa(folder):
    t = time.time()
    print('start', time.time() - t)
    df = pd.read_csv('test_data_idx.csv')

    with open('test_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    df_order = df.groupby('order_id')['pred'].agg(
        {'o_avg': np.mean, 'o_min': np.min, 'o_max': np.max, 'o_cnt': len}).reset_index()
    df_user = df.groupby('user_id')['pred'].agg(
        {'u_avg': np.mean, 'u_min': np.min, 'u_max': np.max, 'u_cnt': len}).reset_index()
    df_item = df.groupby('product_id')['pred'].agg(
        {'p_avg': np.mean, 'p_min': np.min, 'p_max': np.max, 'p_cnt': len}).reset_index()

    df = pd.merge(df, df_order, how='left', on='order_id')
    df = pd.merge(df, df_user, how='left', on='user_id')
    df = pd.merge(df, df_item, how='left', on='product_id').sort_values('order_id')
    data = df[['pred',
               'o_avg', 'o_min', 'o_max', 'o_cnt',
               'u_avg', 'u_min', 'u_max', 'u_cnt',
               'p_avg', 'p_min', 'p_max', 'p_cnt', ]].values
    with open('model2.pkl', 'rb') as f:
        model = pickle.load(f)
    df['pred'] = model.predict_proba(data)[:, 1]
    return df

df = aaa('./')
#df2 = aaa('./only_rebuy/')
#df = df.append(df2)
#df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()


thresh = 0.173
map_result = {}
df = df[['order_id', 'user_id', 'product_id', 'pred']]
for row in df.values:
    order_id, user_id, product_id, pred = row
    order_id = int(order_id)
    if order_id not in map_result:
        map_result[order_id] = []

    if pred > thresh:
        map_result[order_id].append(str(int(product_id)))

f = open('submit.csv', 'w')
f.write('order_id,products\n')
for key in sorted(map_result.keys()):
    val = map_result[key]
    if len(val) == 0:
        val = 'None'
    else:
        val = ' '.join(val)
    f.write('{},{}\n'.format(key, val))
f.close()
