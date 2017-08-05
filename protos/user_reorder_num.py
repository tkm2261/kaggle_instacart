import pandas as pd

df = pd.read_csv('../input/df_prior.csv', usecols=['user_id', 'product_id', 'reordered'])
df = df[df.reordered == 1]

tmp = df.groupby('user_id')['product_id'].apply(lambda x: len((set(x.tolist()))))
tmp = tmp.reset_index()
tmp.columns = ['user_id', 'user_reorder_item_num']
tmp.to_csv('user_reorder_item_num.csv', index=False)

tmp = df.groupby('product_id')['user_id'].apply(lambda x: len((set(x.tolist()))))
tmp = tmp.reset_index()
tmp.columns = ['product_id', 'item_reorder_user_num']
tmp.to_csv('item_reorder_user_num.csv', index=False)

df = pd.read_csv('../input/df_train.csv', usecols=['user_id', 'product_id', 'reordered'])

tmp = df.groupby('product_id')['user_id'].apply(lambda x: len((set(x.tolist()))))
tmp = tmp.reset_index()
tmp.columns = ['product_id', 'item_reorder_user_num_train']
tmp.to_csv('item_reorder_user_num_train.csv', index=False)

tmp = df.groupby('product_id')['reordered'].agg(['count', 'mean', 'sum'])
tmp = tmp.reset_index()
tmp.columns = ['product_id', 'item_reorder_count_train', 'item_reorder_avg_train', 'item_reorder_sum_train']
tmp.to_csv('item_reorder_train.csv', index=False)
