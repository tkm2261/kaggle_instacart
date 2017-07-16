
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


def aaa(arg):
    return f1_score(*arg)


from utils import f1


def get_stack(folder, is_train=True):
    col = 'hogehoge'
    if is_train:
        with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
            df = pd.read_csv(folder + 'train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
            df[col] = pickle.load(f).astype(np.float32)
            df1 = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
    else:
        with open(folder + 'test_tmp.pkl', 'rb') as f:
            df = pd.read_csv(folder + 'test_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
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
    x_train['0714_10000loop'] = get_stack('0714_10000loop/', is_train=True)
    x_train['0715_2nd_order'] = get_stack('0715_2nd_order/', is_train=True)

    fillna_mean = x_train.mean()
    x_train = x_train.fillna(fillna_mean).values.astype(np.float32)

    gc.collect()

    logger.info('load end {}'.format(x_train.shape))

    min_score = (100, 100, 100)
    min_params = None
    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0
    list_idxs = []
    list_model = []
    list_data = []
    for t, (train, test) in enumerate(cv):
        with open('model_%s.pkl' % t, 'rb') as f:
            list_model += [pickle.load(f)]
        list_idxs.append(df.loc[test].reset_index(drop=True).groupby(
            'order_id').apply(lambda x: x.index.values).tolist())
        #list_data.append((trn_x, val_x, trn_y, val_y))

    gc.collect()

    for i in range(7000, 11001, 1000)[:1]:
        list_score = []
        all_pred = np.zeros(y_train.shape[0])
        # for t, (trn_x, val_x, trn_y, val_y) in enumerate(list_data):
        for t, (train, test) in enumerate(cv):
            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test].astype(int)

            list_idx = list_idxs[t]
            clf = list_model[t]
            pred = clf.predict_proba(val_x, num_iteration=i)[:, 1]
            pred = clf.predict_proba(val_x)[:, 1]
            _, _score, _ = f1_metric(val_y, pred, list_idx)
            list_score.append(_score)

        with open('train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('{} {} {}'.format(i, list_score, np.mean(list_score)))
