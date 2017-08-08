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

now_order_ids = None
THRESH = 0.189

list_idx = None
cnt_a = 0


def aaa(arg):
    return f1_score(*arg)


from utils import f1, f1_group  # , f1_group_idx

DIR = 'result_tmp/'


def f1_metric(label, pred):
    res = f1_group(label, pred, list_idx)
    sc = np.mean(res)
    logger.debug('f1: %s' % (sc))
    return 'f1', sc, True


def dummy(label, pred):
    return 'dummy', 0, True


def cst_obj(label, preds):
    label = np.where(label > 0, 1, -1)
    preds = 1.0 / (1.0 + np.exp(-preds))
    #res = f1_group_idx(label, preds, list_idx).astype(np.bool)
    response = - label / (1.0 + np.exp(1.0 + label * preds))
    grad = response
    abs_response = np.fabs(response)
    hess = abs_response * (1 - abs_response)
    #grad[res] *= 0.8
    #grad[~res] *= 1.2
    #grad = preds - label
    #hess = preds * (1.0 - preds)

    return grad, hess


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


def rrr():
    df = pd.read_csv('thresh_target.csv', header=None, names=['order_id', 'user_id', 'product_id', 'reordered'])
    return df


def f1_metric_xgb2(pred, dtrain):
    return 'f12', pred, True


import time


def callback(data):
    if (data.iteration + 1) % 100 != 0:
        return

    clf = data.model
    trn_data = clf.train_set
    val_data = clf.valid_sets[0]
    """
    preds = [ele[2] for ele in clf.eval_train(f1_metric_xgb2) if ele[1] == 'f12'][0]
    labels = trn_data.get_label().astype(np.int)

    res = f1_group_idx(labels, preds, list_idx).astype(np.bool)
    weight = trn_data.get_weight()
    if weight is None:
        weight = np.ones(preds.shape[0])
    weight[res] *= 0.8
    weight[~res] *= 1.25

    trn_data.set_weight(weight)
    """
    preds = [ele[2] for ele in clf.eval_valid(f1_metric_xgb2) if ele[1] == 'f12'][0]
    labels = val_data.get_label().astype(np.int)
    t = time.time()
    res = f1_group(labels, preds, list_idx)
    sc = np.mean(res)

    logger.info('cal [{}] {} {}'.format(data.iteration + 1, sc, time.time() - t))


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    all_params = {'min_child_weight': [10],
                  'subsample': [0.7],
                  'seed': [114514],
                  'n_estimators': [15500],
                  'colsample_bytree': [0.9],
                  'silent': [True],
                  'learning_rate': [0.01],
                  'max_depth': [5],
                  'min_data_in_bin': [8],
                  'min_split_gain': [0],
                  'reg_alpha': [1],
                  'max_bin': [511],
                  #'objective': [cst_obj],
                  #'objective': ['xentropy'],
                  #'metric_freq': [100]
                  }

    with open('fm_data/train_data.pkl', 'rb') as f:
        x_train, y_train, cv = pickle.load(f)
    x_train = x_train.tocsr()
    df = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)        
    logger.info('load end')

    min_score = (100, 100, 100)
    min_params = None
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0

    for params in tqdm(list(ParameterGrid(all_params))):

        cnt = -1
        list_score = []
        list_score2 = []
        list_score3 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv:
            cnt += 1
            # if cnt < 2:
            #    continue
            idx = np.arange(x_train.shape[0], dtype=int)
            _train = idx[train]
            _test = idx[test]            
            trn_x = x_train[_train]
            val_x = x_train[_test]
            trn_y = y_train[train]
            val_y = y_train[test]

            #trn_sc = init_score[train]
            #val_sc = init_score[test]

            list_idx = df.loc[test].reset_index(drop=True).groupby(
                'order_id').apply(lambda x: x.index.values.shape[0]).tolist()
            list_idx = np.array(list_idx, dtype=np.int)

            clf = LGBMClassifier(**params)
            clf.fit(trn_x, trn_y, callbacks=[callback],
                    #init_score=trn_sc, eval_init_score=[val_sc],
                    # sample_weight=trn_w,
                    # eval_sample_weight=[val_w],
                    eval_set=[(val_x, val_y)],
                    verbose=False,
                    eval_metric=dummy,  # f1_metric,
                    # early_stopping_rounds=50
                    )
            pred = clf.predict_proba(val_x)[:, 1]
            all_pred[test] = pred

            _score = log_loss(val_y, pred)
            _score2 = - roc_auc_score(val_y, pred)
            _, _score3, _ = f1_metric(val_y.astype(int), pred.astype(float))
            logger.debug('   _score: %s' % _score3)
            list_score.append(_score)
            list_score2.append(_score2)
            list_score3.append(- 1 * _score3)
            if clf.best_iteration != -1:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])

            with open(DIR + 'train_cv_pred_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(pred, f, -1)
            with open(DIR + 'model_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(clf, f, -1)

            del trn_x
            del clf
            gc.collect()
            break
        with open(DIR + 'train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        params['n_estimators'] = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))
        score3 = (np.mean(list_score3), np.min(list_score3), np.max(list_score3))

        logger.info('param: %s' % (params))
        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('score: {} (avg min max {})'.format(score2[use_score], score2))
        logger.info('score3: {} (avg min max {})'.format(score3[use_score], score2))
        if min_score[use_score] > score3[use_score]:
            min_score = score3
            min_score2 = score2
            min_score3 = score3
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))
        logger.info('best score2: {} {}'.format(min_score2[use_score], min_score2))
        logger.info('best score3: {} {}'.format(min_score3[use_score], min_score3))
        logger.info('best_param: {}'.format(min_params))

    gc.collect()

    # for params in tqdm(list(ParameterGrid(all_params))):
    #    min_params = params
    clf = LGBMClassifier(**min_params)
    clf.fit(x_train, y_train,
            verbose=True,
            eval_metric="auc",
            # sample_weight=sample_weight
            )
    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    ###
    with open(DIR + 'model.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open('fm_data/test_data.pkl', 'rb') as f:
        x_test = pickle.load(f)
    p_test = clf.predict_proba(x_test)
    with open(DIR + 'test_tmp.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)
