import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time
import pickle


def aaa(folder):
    t = time.time()
    print('start', time.time() - t)
    df = pd.read_csv('test_data_idx.csv')

    with open('test_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    return df

df = aaa('./')
#df2 = aaa('./only_rebuy/')
#df = df.append(df2)
#df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()

df['order_id'] = df['order_id'].astype(int)
df_order = df.groupby('order_id')['pred'].agg({'cnt_order': len})
map_cnt_order = df_order.to_dict()['cnt_order']

with open('num_thresh.pkl', 'rb') as f:
    ra, list_thresh = pickle.load(f)


def predict(val, num):
    if np.isnan(val):
        return False
    for i in range(len(ra) - 1):
        if num >= ra[i] and num < ra[i + 1]:
            return val > list_thresh[i]
    raise

map_user_mean = pd.read_csv('../input/user_mean_order.csv', index_col='user_id').to_dict('index')


#thresh = 0.194
map_result = {}
df = df.sort_values(['user_id', 'pred'], ascending=False)
for row in df.values:
    order_id, user_id, product_id, pred = row
    order_id = int(order_id)
    num = map_cnt_order[order_id]
    if order_id not in map_result:
        map_result[order_id] = []

    mean = int(tmp['mean'])
    std = tmp['std']
    th = mean + std
    if predict(pred, num) and len(map_result[order_id]) < th:
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
