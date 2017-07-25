import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import re
from IPython.core.display import display
from tqdm import tqdm as tqdm

df = pd.read_csv('../input/df_prior.csv', usecols=['user_id', 'product_id']).drop_duplicates()

tmp = df.groupby('user_id').apply(lambda x: x.product_id.values.tolist())

prd = sorted(df.product_id.values.tolist())
map_prd = {prd[i]: i for i in range(len(prd))}


def ttt(x):
    a = np.zeros(len(prd), dtype=int)
    idx = [map_prd[i] for i in x]
    a[idx] = 1
    return '{}'.format(a.tolist())


from multiprocessing import Pool

p = Pool()
tmp2 = list(p.map(ttt, tmp.values))
p.close()
p.join()


df2 = pd.DataFrame()
df2['user_id'] = tmp.index.values
df2['pt'] = tmp2

pp = df2.groupby('user_id')['pt'].count()

pp.to_csv('user_item_pattern.csv')
