import pickle
import pandas as pd
from sklearn.metrics import f1_score
import time
t = time.time()
print('start', time.time() - t)
"""
with open('test_data.pkl', 'rb') as f:
    df = pickle.load(f)
    df = df[['o_order_id', 'o_user_id', 'o_product_id']]
    df.columns = ['order_id', 'user_id', 'product_id']
    df.to_csv('test_data_idx.csv', index=False)
"""
df = pd.read_csv('test_data_idx.csv')

with open('test_tmp.pkl', 'rb') as f:
    pred = pickle.load(f)[:, 1]

df['pred'] = pred
thresh = 0.173


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
