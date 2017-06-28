import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time


def aaa(folder):
    t = time.time()
    print('start', time.time() - t)
    df = pd.read_csv('test_data_idx.csv')

    with open('test_tmp2.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    return df

df = aaa('./')
# df2 = aaa('./only_rebuy/')
# df = df.append(df2)
# df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()

map_user_mean = pd.read_csv('../input/user_mean_order.csv', index_col='user_id').to_dict('index')
df = df.sort_values(['order_id', 'pred'], ascending=False)

thresh = 0.194
map_result = {}
for row in df.values:
    order_id, user_id, product_id, pred = row
    order_id = int(order_id)

    tmp = map_user_mean[user_id]
    mean = tmp['mean']
    std = tmp['std']

    if order_id not in map_result:
        map_result[order_id] = []

    if pred > thresh and len(map_result[order_id]) < mean + 0 * std:
        map_result[order_id].append([str(int(product_id)), pred])


def add_none(num, sum_pred):
    score = 2. / (2 + num)
    if sum_pred > score:
        return []
    else:
        print(".", sep="", end="")
        return ['None']


f = open('submit.csv', 'w')
f.write('order_id,products\n')
for key in sorted(map_result.keys()):
    tmp = map_result[key]
    val = [t[0] for t in tmp]
    score = np.sum([t[1] for t in tmp])
    val += add_none(len(val), score)

    if len(val) == 0:
        val = 'None'
    else:
        val = ' '.join(val)
    f.write('{},{}\n'.format(key, val))
f.close()
