import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time
import warnings
warnings.filterwarnings('ignore')
import sys
from tqdm import tqdm
from multiprocessing import Pool

import logging
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s'

logging.basicConfig(format=log_fmt, level=logging.DEBUG)


def multilabel_fscore(y_true, y_pred):
    y_true, y_pred = set(y_true), set(y_pred)
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred)
    recall = tp / len(y_true)

    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def aaa(folder):
    logging.info('enter' + folder)
    # with open(folder + 'train_cv_pred.pkl', 'rb') as f:
    #    pred = pickle.load(f)
    with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)

    df = pd.read_csv(folder + 'train_data_idx.csv')
    df.drop('target', axis=1, inplace=True)
    df_val = df  # .loc[val, :].copy()

    df_val['pred'] = pred
    logging.info('exit')
    return df_val


def make_result():
    logging.info('enter')
    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id', 'reordered'])
    # df = df[df['user_id'].isin(test)].copy()

    map_result = {}
    for row in df[['order_id', 'user_id', 'product_id', 'reordered']].values:
        order_id, user_id, product_id, reordered = row
        if reordered == 0:
            continue
        order_id = int(order_id)
        if order_id not in map_result:
            map_result[order_id] = []

        map_result[order_id].append(int(product_id))
    logging.info('exit')
    return map_result


logging.info('user mean')
map_user_mean = pd.read_csv('../input/user_mean_order.csv', index_col='user_id').to_dict('index')
map_result = make_result()

#df_val = aaa('./').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val1 = aaa('./0705_new/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val.pred += df_val1.pred.values
#df_val = aaa('./0710_stack2/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
df_val = aaa('./0705_old_rate001/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val = aaa('./0706_tuned/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val1 = aaa('./0708_gpu_ids/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
#df_val2 = aaa('./0708_ids/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val.pred += df_val1.pred.values
# df_val = aaa('./0707_stack/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
# df_val.pred += df_val1.pred.values
# df_val1 = aaa('./0705_new_rate001/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)

#df_val.pred = np.mean(np.vstack([df_val.pred.values, df_val1.pred.values]), axis=0)
# df_val.pred = np.max(np.vstack([df_val.pred.values, df_val2.pred.values]), axis=0)


df_val = df_val.sort_values(['order_id', 'pred'], ascending=False)
df_val = df_val[['order_id', 'user_id', 'product_id', 'pred']].values

map_pred = {}
n = df_val.shape[0]
for i in tqdm(range(n)):
    order_id, user_id, product_id, pred = df_val[i]
    order_id, user_id, product_id = list(map(int, [order_id, user_id, product_id]))

    tmp = map_user_mean[user_id]
    mean = tmp['mean']
    std = tmp['std']
    if order_id not in map_pred:
        map_pred[order_id] = []
    map_pred[order_id].append((product_id, pred, mean, std, user_id))


with open('item_info.pkl', 'wb') as f:
    pickle.dump((map_pred, map_result), f, -1)


with open('item_info.pkl', 'rb') as f:
    map_pred, map_result = pickle.load(f)


def add_none(num, sum_pred, safe):
    score = 2. / (2 + num)
    if sum_pred * safe > score:
        return False
    else:
        return True


np.random.seed(0)
NUM = 10000


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


ALPHA = 0.3  # float(sys.argv[1])
THRESH_NUM = int(sys.argv[1])
logging.info('ALPHA: %s' % ALPHA)
logging.info('THRESH_NUM: %s' % THRESH_NUM)


def get_y_true2(preds, none_idx, cov_matrix):
    n = preds.shape[0]

    cov_matrix = ALPHA * cov_matrix + (1 - ALPHA) * np.eye(n)

    tmp = np.random.multivariate_normal(np.zeros(n), cov_matrix, size=NUM)
    preds = np.array([norm.ppf(q=p, loc=0, scale=np.sqrt(cov_matrix[i, i])) for i, p in enumerate(preds)])

    y_true = preds > tmp
    y_true_sum = y_true.sum(axis=1)
    y_true[:, none_idx] = np.where(y_true_sum == 0, True, False)
    return y_true


IS_COV = False
with open('map_user_order_num.pkl', 'rb') as f:
    map_user_order_num = pickle.load(f)

with open('map_reoder_rate.pkl', 'rb') as f:
    map_reoder_rate = pickle.load(f)


df_a = pd.read_csv('../input/df_train.csv', usecols=['product_id', 'reordered'], dtype=int)
# set_product = set(df_a[df_a['reordered'] == 1]['product_id'].unique().tolist())
set_product = set(df_a['product_id'].unique().tolist())


def _uuu(args):
    order_id, vals = args
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    items = [int(product_id) for product_id, _, _, _, _ in vals]
    user_id = vals[0][4]
    mean = vals[0][2]
    std = vals[0][3]

    if IS_COV:

        n = preds.shape[0]
        if map_user_order_num[user_id] >= THRESH_NUM:
            cov_data = get_cov(user_id)
            idx = [cov_data.map_item2idx[i] for i in items]
            try:
                tmp = cov_data.cov_matrix[idx, idx]
            except:
                tmp = np.array([[1]])
            cov_matrix = np.zeros((n + 1, n + 1))
            cov_matrix[:n, :n] += tmp
            cov_matrix[n - 1, n - 1] = 1
        else:
            cov_matrix = np.eye(n + 1)

    none_prob = (1 - preds).prod()  # min(1 - map_reoder_rate[user_id], (1 - preds).prod())
    preds = np.r_[preds, [none_prob]]
    items.append('None')

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]

    items = [items[i] for i in idx]  # items[idx]
    none_idx = idx[-1]
    # sum_pred = preds.sum()
    """
    if IS_COV:
        scenario = get_y_true2(preds, none_idx, cov_matrix)
    else:
        scenario = get_y_true(preds, none_idx)
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
    """
    """
    scores = []
    tp = 0
    ans = map_result.get(order_id, ['None'])
    for i in range(len(preds)):
        f1 = multilabel_fscore(ans, items[:i + 1])
        scores.append((f1, i))
    _, best_idx = max(scores, key=lambda x: x[0])
    """
    scores = []
    num_y_true = preds.sum()
    tp = 0
    ans = map_result.get(order_id, ['None'])
    for i in range(len(preds)):
        tp += preds[i]
        precision = tp / (i + 1)
        recall = tp / num_y_true
        f1 = (2 * precision * recall) / (precision + recall)
        scores.append((f1, i))
    f1, idx = max(scores, key=lambda x: x[0])
    # idx = int(np.around(map_order_cv_final[order_id]))
    idx = max(0, idx)
    idx = min(len(items) - 1, idx)
    score = items[:idx + 1]
    """
    label = best_idx
    data = [f1, idx, len(items), preds[idx], preds[min(idx + 1, len(items) - 1)], preds.sum(), preds.mean(), preds.min(), preds.max(),
            mean, std, map_user_order_num[user_id], map_reoder_rate[user_id]]
    data = [order_id, user_id, label] + data
    str_data = ",".join(map(str, data))
    with open('final_data/%s.csv' % order_id, 'w') as f:
        f.write(str_data + '\n')
    """
    """
    if add_none(idx + 1, f1, 1):
        if 'None' not in score:
            score += ['None']
    """
    ans = map_result.get(order_id, ['None'])
    sc = multilabel_fscore(ans, score)
    # print(ans, score, f1, sc)
    return sc


def uuu(args):
    order_id, vals = args
    items = [int(product_id) for product_id, _, _, _, _ in vals]
    # preds = np.array([pred_val if items[i] in set_product else 0 for i, (_, pred_val, _, _, _) in enumerate(vals)])
    preds = np.array([pred_val for _, pred_val, _, _, _ in vals])
    user_id = vals[0][4]

    none_prob = (1 - preds).prod()  # min(1 - map_reoder_rate[user_id], (1 - preds).prod())
    preds = np.r_[preds, [none_prob]]
    items.append('None')

    idx = np.argsort(preds)[::-1]
    preds = preds[idx]

    items = [items[i] for i in idx]  # items[idx]
    none_idx = idx[-1]
    idxs = []
    _preds = preds.copy()
    for _ in range(1):
        scores = []
        num_y_true = preds.sum()
        tp = 0
        ans = map_result.get(order_id, ['None'])

        for i in range(len(preds)):
            tp += preds[i]
            precision = tp / (i + 1)
            recall = tp / num_y_true
            f1 = (2 * precision * recall) / (precision + recall)
            scores.append((f1, i))

        f1, idx = max(scores, key=lambda x: x[0])
        idxs.append(idx)

    # idx = int(np.around(np.mean(idxs)))
    score = items[:idx + 1]
    ans = map_result.get(order_id, ['None'])
    sc = multilabel_fscore(ans, score)
    # print(ans, score, f1, sc)
    return sc


def sss2(map_pred):
    res = []
    p = Pool(7)
    aaaa = sorted(map_pred.items(), key=lambda x: x[0])
    res = p.map(uuu, tqdm(aaaa))
    # res = list(map(uuu, tqdm(aaaa)))
    p.close()
    p.join()
    return np.array(res)  # np.mean(res)


logging.info('start')
# idx = pd.read_csv('tmp_use.csv', header=None)[0].values
sc = sss2(map_pred)
score = np.mean(sc)
print(score, np.mean(sc))
# pd.DataFrame({174: rrr[0], 194: rrr[1]}).to_csv('tmp.csv', index=False)
logging.info('end')
