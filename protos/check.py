import re
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score, f1_score
import gc
from logging import getLogger
logger = getLogger(None)
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

from load_data import load_train_data, load_test_data
from features_drop import DROP_FEATURE
from features_use import FEATURE
now_order_ids = None
THRESH = 0.189

DIR = 'result_tmp/'


def aaa(arg):
    return f1_score(*arg)


from utils import f1


def get_stack(folder, is_train=True):
    col = 'hogehoge'
    if is_train:
        with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
            df = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
            df[col] = pickle.load(f).astype(np.float32)
            df1 = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
    else:
        with open(folder + 'test_tmp.pkl', 'rb') as f:
            df = pd.read_csv('test_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
            df[col] = pickle.load(f).astype(np.float32)[:, 1]
            df1 = pd.read_csv('test_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)

    return pd.merge(df1, df, how='left', on=['order_id', 'user_id', 'product_id'])[col].values


def f1_metric(label, pred, list_idx):
    res = [f1(label[i], pred[i]) for i in list_idx]
    sc = np.mean(res)
    return 'f1', sc, True


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler('check.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    df = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
    ###
    x_train, y_train, cv = load_train_data()
    logger.info('merges')
    x_train = x_train.merge(pd.read_csv('product_last.csv').astype(np.float32).rename(columns={'product_id': 'o_product_id'}), how='left',
                            on='o_product_id', copy=True)
    x_train = x_train.merge(pd.read_csv('product_first.csv').astype(np.float32).rename(columns={'product_id': 'o_product_id'}), how='left',
                            on='o_product_id', copy=True)
    x_train = x_train.merge(pd.read_csv('product_all.csv').astype(np.float32).rename(columns={'product_id': 'o_product_id'}), how='left',
                            on='o_product_id', copy=True)
    x_train = x_train.merge(pd.read_csv('word_preds.csv').astype(np.float32).rename(columns={'product_id': 'o_product_id'}), how='left',
                            on='o_product_id', copy=True)

    x_train = x_train.merge(pd.read_csv('user_item_pattern.csv').astype(np.float32).rename(columns={'user_id': 'o_user_id'}), how='left',
                            on='o_user_id', copy=True)
    x_train['stack1'] = get_stack('result_0727/')

    id_cols = [col for col in x_train.columns.values
               if re.search('_id$', col) is not None and
               col not in set(['o_user_id', 'o_product_id', 'p_aisle_id', 'p_department_id'])]
    logger.debug('id_cols {}'.format(id_cols))
    x_train.drop(id_cols, axis=1, inplace=True)

    dropcols = sorted(list(set(x_train.columns.values.tolist()) & set(DROP_FEATURE)))
    x_train.drop(dropcols, axis=1, inplace=True)
    fillna_mean = x_train.mean()
    x_train = x_train.fillna(fillna_mean).values.astype(np.float32)

    gc.collect()

    logger.info('load end {}'.format(x_train.shape))

    min_score = (100, 100, 100)
    min_params = None
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0
    list_idxs = []
    list_model = []
    list_data = []
    for t, (train, test) in enumerate(cv[: 1]):
        with open(DIR + 'model_%s.pkl' % t, 'rb') as f:
            list_model += [pickle.load(f)]
        list_idxs.append(df.loc[test].reset_index(drop=True).groupby(
            'order_id').apply(lambda x: x.index.values).tolist())
        # list_data.append((trn_x, val_x, trn_y, val_y))

    gc.collect()

    for i in range(500, 11001, 100):
        list_score = []
        all_pred = np.zeros(y_train.shape[0])
        # for t, (trn_x, val_x, trn_y, val_y) in enumerate(list_data):
        for t, (train, test) in enumerate(cv[:1]):
            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test].astype(int)

            list_idx = list_idxs[t]
            clf = list_model[t]
            pred = clf.predict_proba(val_x, num_iteration=i)[:, 1]
            _, _score, _ = f1_metric(val_y, pred, list_idx)
            list_score.append(_score)

        # with open('train_cv_tmp.pkl', 'wb') as f:
        #    pickle.dump(all_pred, f, -1)

        logger.info('{} {} {}'.format(i, list_score, np.mean(list_score)))
