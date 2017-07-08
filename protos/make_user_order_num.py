import pandas as pd
import pickle
df = pd.read_csv('../input/orders.csv', usecols=['order_id', 'user_id', 'eval_set'])
df = df[df['eval_set'] == 'prior']
a = df.groupby('user_id').apply(len)
a = a.to_dict()

with open("map_user_order_num.pkl", "wb") as f:
    pickle.dump(a, f, -1)
