import recsys.algorithm
import pandas as pd
from tqdm import tqdm
recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD
svd = SVD()
svd.load_data(filename='../input/user_item_cnt_noheader.csv', sep=',', format={'col':1, 'row':0, 'value':2, 'ids': int})

k = 100
svd.compute(k=k,
            min_values=1,
            pre_normalize=None,
            mean_center=True,
            post_normalize=True,
            savefile='./tmp')

users = pd.read_csv('../input/user_item_cnt.csv', usecols=['user_id'])['user_id'].unique()

for user_id in tqdm(user):
    ret = svd.recommend(user_id, 100, is_row=False)
    import pdb;pdb.set_trace()
