import pickle
import pandas as pd
from sklearn.metrics import f1_score
import time


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


thresh = 0.176
map_result = {}
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
