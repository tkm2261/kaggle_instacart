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
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score, f1_score
import gc
from logging import getLogger
logger = getLogger(None)
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

now_order_ids = None
THRESH = 0.189

list_idx = None


def aaa(arg):
    return f1_score(*arg)


from utils import f1


def f1_metric(label, pred):
    res = [f1(label.take(i), pred.take(i)) for i in list_idx]
    sc = np.mean(res)
    return 'f1', sc, True


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


def rrr():
    df = pd.read_csv('thresh_target.csv', header=None, names=['order_id', 'user_id', 'product_id', 'reordered'])
    return df


def hoge(data):
    weight = np.arange(1000)
    clf = data.model
    trn_data = clf.train_set
    y_train = np.random.random(1000) > 0.8
    #val_data = clf.valid_sets[0]
    # trn_data.set_weight(weight)
    trn_data.set_field('weight', weight)

    # trn_data.set_label(y_train)
    # print(trn_data.get_weight())
    clf.set_train_set(trn_data)


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler('train2.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    all_params = {'max_depth': [5],
                  'learning_rate': [0.1],  # [0.06, 0.1, 0.2],
                  #'n_estimators': [3000],
                  'min_data_in_leaf': [10],
                  'feature_fraction': [0.9],
                  #'boosting_type': ['dart'],  # ['gbdt'],
                  #'xgboost_dart_mode': [False],
                  #'num_leaves': [96],
                  'metric': ['binary_logloss'],
                  'objective': ['binary'],  # , 'xentropy'],
                  'bagging_fraction': [0.7],
                  #'min_child_samples': [10],
                  'lambda_l1': [1],
                  #'lambda_l2': [1],
                  'max_bin': [511],
                  'min_split_gain': [0],
                  #'device': ['gpu'],
                  #'gpu_platform_id': [0],
                  #'gpu_device_id': [0],
                  'verbose': [0],
                  'seed': [6436]
                  }

    logger.info('load start')
    np.random.seed(0)
    x_train = np.random.random((1000, 100))
    y_train = np.random.random(1000) > 0.8
    logger.info('load end {}'.format(x_train.shape))

    min_score = (100, 100, 100)
    min_params = None
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0
    weight = np.arange(1000)

    for params in tqdm(list(ParameterGrid(all_params))):
        train_data = lgb.Dataset(x_train, label=y_train)
        # test_data = lgb.Dataset(val_x, label=val_y)
        clf = lgb.train(params,
                        train_data,
                        10,  # params['n_estimators'],
                        callbacks=[hoge],
                        # early_stopping_rounds=30,
                        valid_sets=[train_data],
                        # feval=f1_metric_xgb,
                        # fobj=logregobj
                        )
