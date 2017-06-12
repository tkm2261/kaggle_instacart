import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('../input/df_train.csv', usecols=['user_id', 'product_id'])
df = df.groupby('user_id').count()
df['product_id'] = (df['product_id'] / 10).astype(int)
df.loc[df['product_id'] > 5, 'product_id'] = 5

df = df.reset_index(drop=False)

x_train = df.values
y_train = df['product_id'].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)

list_cv = []
for train, test in cv.split(x_train, y_train):
    trn_user_id = x_train[train, 0]
    val_user_id = x_train[test, 0]
    list_cv.append((trn_user_id, val_user_id))

with open('user_split.pkl', 'wb') as f:
    pickle.dump(list_cv, f, -1)
