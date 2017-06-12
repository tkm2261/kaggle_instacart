import networkx as nx
import pandas as pd
from tqdm import tqdm

df_prior = pd.read_csv('../input/df_prior.csv', usecols=['order_id', 'product_id'])
df = df_prior.groupby('order_id')['product_id'].agg(lambda x: " ".join(map(str, x.tolist())))
df.to_csv('item_corpus.csv', index=False, header=False)
