import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time


def multilabel_fscore(y_true, y_pred):
    """
    ex1:
    y_true = [1, 2, 3]
    y_pred = [2, 3]
    return: 0.8

    ex2:
    y_true = ["None"]
    y_pred = [2, "None"]
    return: 0.666

    ex3:
    y_true = [4, 5, 6, 7]
    y_pred = [2, 4, 8, 9]
    return: 0.25

    """
    y_true, y_pred = set(y_true), set(y_pred)
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred)
    recall = tp / len(y_true)

    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def aaa(folder):
    t = time.time()
    print('start', time.time() - t)
    df = pd.read_csv('test_data_idx.csv')

    with open('test_tmp2.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    return df

df = aaa('./')
# df2 = aaa('./only_rebuy/')
# df = df.append(df2)
# df = df.groupby(['order_id', 'product_id', 'user_id']).max().reset_index()

df = df.sort_values(['order_id', 'pred'], ascending=False)

thresh = 0.194
map_result = {}
for row in df.values:
    order_id, user_id, product_id, pred = row
    order_id = int(order_id)

    if order_id not in map_result:
        map_result[order_id] = []

    if pred > thresh:
        map_result[order_id].append([int(product_id), pred])


from tqdm import tqdm


def get_y_true(vals):
    y_true = [product_id
              for product_id, pred_val in vals if pred_val > np.random.uniform()]
    if len(y_true) == 0:
        y_true = ['None']

    return y_true

from multiprocessing import Pool


def uuu(args):
    order_id, vals = args

    sum_pred = sum(pred_val for _, pred_val in vals)
    if sum_pred < 1:
        vals += [('None', 1 - sum_pred)]
        vals = sorted(vals, key=lambda x: x[1], reverse=True)
    items = [product_id for product_id, _ in vals]
    scenario = [get_y_true(vals) for _ in range(1000)]

    scores = []
    for i in range(len(vals)):
        pred = items[:i + 1]
        f1 = np.mean([multilabel_fscore(sc, pred) for sc in scenario])
        scores.append((f1, pred))

    f1, items = max(scores, key=lambda x: x[0])
    return order_id, items

#p = Pool()
result = list(map(uuu, tqdm(map_result.items())))
# p.close()
# p.join()

f = open('submit.csv', 'w')
f.write('order_id,products\n')
for key, val in sorted(result, key=lambda x: x[0]):
    val = " ".join(map(str, val))
    f.write('{},{}\n'.format(key, val))
f.close()
