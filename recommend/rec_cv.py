import pickle
import pandas as pd
from sklearn.metrics import f1_score
import time


def aaa():
    t = time.time()
    df = pd.read_csv('svd2.csv')
    df.columns = ['dist', 'product_id', 'user_id']
    print('start2', time.time() - t)
    with open('../protos/user_split.pkl', 'rb') as f:
        cv = pickle.load(f)

    list_cv = []
    user_ids = df['user_id']

    for train, test in cv[:1]:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)
        list_cv.append((trn, val))

    df_val = df.loc[val, :].copy()
    print('start3', time.time() - t)

    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id'])
    df = df[df['user_id'].isin(test)].copy()
    df['target'] = 1

    print('start4', time.time() - t)
    df = pd.merge(df, df_val, how='outer', on=['user_id', 'product_id'])
    print('start5', time.time() - t)
    return df

df = aaa()
print('size', df.shape)
#df = aaa('./only_rebuy/')
#df = df.append(df2)
#df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()

alldf = pd.read_csv('svd.csv')

for thresh in range(101):
    thresh /= 100
    df['pred_label'] = df['dist'] < thresh
    alldf['pred_label'] = alldf['dist'] < thresh

    df['label'] = df['target'] == 1
    df['aaa'] = (df['label'] == 1) & (df['pred_label'] == 1)

    print(thresh, f1_score(df.label.values, df.pred_label.values), df[
          'pred_label'].sum(), alldf['pred_label'].sum(), df['aaa'].sum())
