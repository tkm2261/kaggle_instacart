import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

import time
from tqdm import tqdm
import logging
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s'

logging.basicConfig(format=log_fmt, level=logging.DEBUG)


def aaa(folder):
    t = time.time()
    print('start', time.time() - t)
    df = pd.read_csv('test_data_idx.csv')
    df.sort_values(['order_id', 'user_id', 'product_id'], inplace=True)

    with open(folder + 'test_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    return df


df = aaa('./result_0728_18000/')
# df2 = aaa('./only_rebuy/')
# df = df.append(df2)
# df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()

df = df.sort_values(['order_id', 'pred'], ascending=False)
df = df[['order_id', 'user_id', 'product_id', 'pred']].values

map_user_mean = pd.read_csv('../input/user_mean_order.csv', index_col='user_id').to_dict('index')

map_pred = {}
n = df.shape[0]
for i in tqdm(range(n)):
    order_id, user_id, product_id, pred = df[i]
    order_id, user_id, product_id = list(map(int, [order_id, user_id, product_id]))

    tmp = map_user_mean[user_id]
    mean = tmp['mean']
    std = tmp['std']
    if order_id not in map_pred:
        map_pred[order_id] = []
    map_pred[order_id].append((product_id, pred, mean, std, user_id))

np.random.seed(0)

from multiprocessing import Pool

np.random.seed(0)
NUM = 20000
logging.info('NUM: {}'.format(NUM))

ALPHA = 0.2
IS_COV = 0
logging.info('ALPHA: {}, IS_COV: {}'.format(ALPHA, IS_COV))

#ALPHA, IS_COV = sys.argv[1:3]
#ALPHA = float(ALPHA)
#IS_COV = int(IS_COV)

ALPHA2 = 0.4 #sys.argv[1]
ALPHA2 = float(ALPHA2)
logging.info('ALPHA2: {}'.format(ALPHA2))

with open('map_user_order_num.pkl', 'rb') as f:
    map_user_order_num = pickle.load(f)

with open('map_reoder_rate.pkl', 'rb') as f:
    map_reoder_rate = pickle.load(f)


df_a = pd.read_csv('../input/df_train.csv', usecols=['product_id', 'reordered'], dtype=int)
# set_product = set(df_a[df_a['reordered'] == 1]['product_id'].unique().tolist())
set_product = set(df_a['product_id'].unique().tolist())


def get_cov(user_id, items):
    n = len(items)
    with open('../recommend/cov_data/%s.pkl' % user_id, 'rb') as f:
        cov_data = pickle.load(f)
    idx = [cov_data.map_item2idx[i] for i in items]
    try:
        tmp = all_cov_data.cov_matrix[idx][:, idx]
    except:
        tmp = np.array([[1]])
    cov_matrix = np.zeros((n + 1, n + 1))
    cov_matrix[:n, :n] += tmp
    cov_matrix[n - 1, n - 1] = 1
    return cov_matrix


from scipy.stats import norm
logging.info('all cov')
with open('../recommend/all_cov.pkl', 'rb') as f:
    all_cov_data = pickle.load(f)
logging.info('all cov end')


def get_all_cov(user_id, items):
    n = len(items)
    idx = [all_cov_data.map_item2idx[i] for i in items]
    try:
        tmp = all_cov_data.cov_matrix[idx][:, idx]
    except:
        tmp = np.array([[1]])
    cov_matrix = np.zeros((n + 1, n + 1))
    cov_matrix[:n, :n] += tmp
    cov_matrix[n - 1, n - 1] = 1
    return cov_matrix


def get_y_true2(preds, none_idx, cov_matrix):
    n = preds.shape[0]
    tmp = np.random.multivariate_normal(np.zeros(n), cov_matrix, size=NUM)
    preds = np.array([norm.ppf(q=p, loc=0, scale=np.sqrt(cov_matrix[i, i])) for i, p in enumerate(preds)])

    y_true = preds > tmp
    y_true_sum = y_true.sum(axis=1)
    y_true[:, none_idx] = np.where(y_true_sum == 0, True, False)
    return y_true

        
def expect(preds, items):    
    none_prob = (1 - preds).prod()
    preds = np.r_[preds, [none_prob]]
    items.append('None')

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]
    items = [items[i] for i in idx]
    
    scores = []
    num_y_true = preds.sum()
    tp = 0

    for i in range(len(preds)):
        tp += preds[i]
        precision = tp / (i + 1)
        recall = tp / num_y_true
        f1 = (2 * precision * recall) / (precision + recall)
        scores.append((f1, i))

    f1, idx = max(scores, key=lambda x: x[0])
    score = items[:idx + 1]
    return score

    
def sampling(preds, items, user_id, alpha, is_all_conv=True):
    n = preds.shape[0]
    if is_all_conv:
        cov_matrix = get_cov(user_id, items) * alpha + (1 - alpha) * get_all_cov(user_id, items) #np.eye(n + 1)
    else:
        cov_matrix = get_cov(user_id, items) * alpha + (1 - alpha) * np.eye(n + 1)

    none_prob = (1 - preds).prod()
    preds = np.r_[preds, [none_prob]]
    items.append('None')
        
    idx = np.argsort(preds)[::-1]
    preds = preds[idx]

    items = [items[i] for i in idx]
    none_idx = idx[-1]
    cov_matrix = cov_matrix[idx][:, idx]
    
    scenario = get_y_true2(preds, none_idx, cov_matrix)

    num_y_true = scenario.sum(axis=1)
    scores = []
    tp = np.zeros(scenario.shape[0])
    for i in range(len(preds)):
        num_y_pred = i + 1
        # tp = scenario[:, :i + 1].sum(axis=1)
        tp += scenario[:, i]
        precision = tp / num_y_pred
        recall = tp / num_y_true
        f1 = (2 * precision * recall) / (precision + recall)
        f1[np.isnan(f1)] = 0
        f1 = f1.mean()
        scores.append((f1, i))
    f1, idx = max(scores, key=lambda x: x[0])
    score = items[:idx + 1]

    return score

def uuu(args):
    order_id, vals = args
    
    items = [int(product_id) for product_id, _, _, _, _ in vals]
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    user_id = vals[0][4]
    n = preds.shape[0]
    try:
        if map_user_order_num[user_id] >= 10:
            score = sampling(preds, items, user_id, alpha=ALPHA, is_all_conv=IS_COV)
        else:
            score = sampling(preds, items, user_id, alpha=ALPHA2, is_all_conv=0)
    except:
        score = expect(preds, items)
        
    return order_id, score


p = Pool()
#result = list(map(uuu, tqdm(map_pred.items())))
result = list(p.map(uuu, tqdm(map_pred.items())))
p.close()
p.join()

f = open('submit.csv', 'w')
f.write('order_id,products\n')
for key, val in sorted(result, key=lambda x: x[0]):
    val = " ".join(map(str, val))
    f.write('{},{}\n'.format(key, val))
f.close()
