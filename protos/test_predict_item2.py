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
    df = pd.read_csv(folder + 'test_data_idx.csv')

    with open(folder + 'test_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    return df


df = aaa('./0705_old_rate001/')
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
NUM = 1000


def get_y_true(preds, none_idx):
    n = preds.shape[0]
    y_true = np.zeros(n, dtype=np.bool)
    y_true = np.random.random((NUM, n)) < preds

    y_true_sum = y_true.sum(axis=1)
    y_true[:, none_idx] = np.where(y_true_sum == 0, True, False)

    return y_true


from scipy.stats import norm


def get_cov(user_id):
    with open('../recommend/cov_data/%s.pkl' % user_id, 'rb') as f:
        return pickle.load(f)


ALPHA = 0.  # float(sys.argv[1])
logging.info('ALPHA: %s' % ALPHA)


def get_y_true2(preds, none_idx, cov_matrix):
    n = preds.shape[0]

    cov_matrix = ALPHA * cov_matrix + (1 - ALPHA) * np.eye(n)

    tmp = np.random.multivariate_normal(np.zeros(n), cov_matrix, size=NUM)
    preds = np.array([norm.ppf(q=p, loc=0, scale=np.sqrt(cov_matrix[i, i])) for i, p in enumerate(preds)])

    y_true = preds > tmp
    y_true_sum = y_true.sum(axis=1)
    y_true[:, none_idx] = np.where(y_true_sum == 0, True, False)
    return y_true


from multiprocessing import Pool


def uuu(args):
    order_id, vals = args

    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    items = [int(product_id) for product_id, _, _, _, _ in vals]

    user_id = vals[0][4]

    cov_data = get_cov(user_id)
    idx = [cov_data.map_item2idx[i] for i in items]
    try:
        tmp = cov_data.cov_matrix[idx, idx]
    except:
        tmp = np.array([[1]])
    n = preds.shape[0]
    cov_matrix = np.zeros((n + 1, n + 1))
    cov_matrix[:n, :n] += tmp
    cov_matrix[n - 1, n - 1] = 1

    #none_prob = max(1 - preds.sum(), 0) #
    none_prob = (1 - preds).prod()
    preds = np.r_[preds, [none_prob]]
    items.append('None')

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]

    items = [items[i] for i in idx]  # items[idx]
    none_idx = idx[-1]
    # sum_pred = preds.sum()
    # scenario = get_y_true(preds, none_idx)  # np.array([get_y_true(preds, none_idx) for _ in range(1000)])
    scenario = get_y_true2(preds, none_idx, cov_matrix)
    num_y_true = scenario.sum(axis=1)
    scores = []
    tp = np.zeros(scenario.shape[0])
    for i in range(len(preds)):
        num_y_pred = i + 1
        tp += scenario[:, i]
        precision = tp / num_y_pred
        recall = tp / num_y_true
        f1 = (2 * precision * recall) / (precision + recall)
        f1[np.isnan(f1)] = 0
        f1 = f1.mean()
        scores.append((f1, i))
    f1, idx = max(scores, key=lambda x: x[0])
    score = items[:idx + 1]

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
