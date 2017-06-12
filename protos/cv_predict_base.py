import pickle
import pandas as pd
from sklearn.metrics import f1_score
import time
t = time.time()
print('start', time.time() - t)
with open('train_cv_pred_base.pkl', 'rb') as f:
    pred = pickle.load(f)

print('start1', time.time() - t)
with open('train_baseline.pkl', 'rb') as f:
    df = pickle.load(f)[0]
    df = df[['order_id', 'user_id', 'product_id']]
    df.to_csv('train_data_idx_base.csv', index=False)
"""
df = pd.read_csv('train_data_idx_base.csv')
"""
print('start2', time.time() - t)
with open('user_split.pkl', 'rb') as f:
    cv = pickle.load(f)

list_cv = []
user_ids = df['user_id']


for train, test in cv[:1]:
    trn = user_ids.isin(train)
    val = user_ids.isin(test)
    list_cv.append((trn, val))

df_val = df.loc[val, :].copy()
df_val['pred'] = pred
print('start3', time.time() - t)

df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id'])
df = df[df['user_id'].isin(test)].copy()
df['target'] = 1

print('start4', time.time() - t)
df = pd.merge(df, df_val, how='outer', on=['order_id', 'user_id', 'product_id'])
print('start5', time.time() - t)

thresh = 0.128

for thresh in range(10, 20):
    thresh /= 100
    df['pred_label'] = df['pred'] > thresh
    df['label'] = df['target'] == 1

    print(thresh, f1_score(df.label.values, df.pred_label.values))
