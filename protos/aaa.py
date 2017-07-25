import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import re
from IPython.core.display import display
from tqdm import tqdm as tqdm

df = pd.read_csv('../input/df_prior_30.csv.gz', usecols=['user_id', 'department_id', 'reordered']).drop_duplicates()
df = df[df.reordered == 1]
tmp = df.groupby('user_id').apply(lambda x: x.department_id.values.tolist())


def ttt(x):
    return ','.join(map(str, sorted(x)))


from multiprocessing import Pool
print(ttt(tmp.values[0]))
p = Pool()
tmp2 = list(p.map(ttt, tqdm(tmp.values)))
p.close()
p.join()

df2 = pd.DataFrame()
df2['user_id'] = tmp.index.values
df2['pt'] = tmp2

pp = df2.groupby('pt')['user_id'].count().reset_index()
pp.columns = ['pt', 'pt_count']

df2 = df2.merge(pp, how='left', on='pt')
df2[['user_id', 'pt_count']].to_csv('user_depart_pattern.csv', index=False)
