import pandas as pd
import pickle
df = pd.read_csv('../input/df_prior.csv', usecols=['order_id', 'user_id', 'reordered'])
df = df.groupby('order_id').agg({'reordered': 'sum', 'user_id': 'max'})
df = df.groupby('user_id')['reordered'].mean().to_dict()

with open("map_reoder_rate.pkl", "wb") as f:
    pickle.dump(df, f, -1)
