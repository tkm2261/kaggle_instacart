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
import lightgbm as lgb
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

from utils import f1, f1_group, f1_group_idx

DIR = 'result_tmp/'


def f1_metric(_pred, dtrain):
    pred = 1.0 / (1.0 + np.exp(- _pred))
    pred = pred.astype(np.float)

    label = dtrain.get_label().astype(np.int)

    res = f1_group(label, pred, list_idx)

    sc = np.mean(res)
    logger.debug('f1: %s' % (sc))
    return 'f1', sc, True


def logregobj(_preds, dtrain):

    labels = dtrain.get_label().astype(np.int)
    preds = (1.0 / (1.0 + np.exp(- _preds))).astype(np.float)

    #res = f1_group_idx(labels, preds, list_idx).astype(np.bool)

    grad = preds - labels
    hess = preds * (1.0 - preds)
    """
    grad[res] = grad[res] * 1.1
    grad[~res] = grad[~res] * 0.9

    hess[res] = hess[res] * 1.1
    hess[~res] = hess[~res] * 0.9
    """
    return grad, hess


def f1_metric_xgb(pred, dtrain):
    label = dtrain.get_label().astype(np.int)
    pred = pred.astype(np.float64)
    #res = [f1(label.take(i), pred.take(i)) for i in list_idx]
    res = f1_group(label, pred, list_idx)
    sc = np.mean(res)
    logger.debug('f1: %s' % (sc))
    return 'f1', - sc


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


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler('train.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('load start')

    x_train, y_train, cv = load_train_data()

    df = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)

    weight_col = 'order_id'
    order_idx = df.order_id.values
    tmp = df.groupby(weight_col)[[weight_col]].count()
    tmp.columns = ['weight']
    df = pd.merge(df, tmp.reset_index(), how='left', on=weight_col)
    sample_weight = 1 / np.log(df['weight'].values)
    # sample_weight *= (sample_weight.shape[0] / sample_weight.sum())

    logger.info('merges')
    x_train['stack1'] = get_stack('result_0727/')
    id_cols = [col for col in x_train.columns.values
               if re.search('_id$', col) is not None and
               col not in set(['o_user_id', 'o_product_id', 'p_aisle_id', 'p_department_id'])]
    logger.debug('id_cols {}'.format(id_cols))
    x_train.drop(id_cols, axis=1, inplace=True)

    dropcols = sorted(list(set(x_train.columns.values.tolist()) & set(DROP_FEATURE)))
    x_train.drop(dropcols, axis=1, inplace=True)

    usecols = x_train.columns.values
    #logger.debug('all_cols {}'.format(usecols))
    with open('usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)
    gc.collect()

    fillna_mean = x_train.mean()
    with open('fillna_mean.pkl', 'wb') as f:
        pickle.dump(fillna_mean, f, -1)

    x_train = x_train.fillna(fillna_mean).values.astype(np.float32)

    logger.info('data end')
    # x_train[np.isnan(x_train)] = -10
    gc.collect()

    logger.info('load end {}'.format(x_train.shape))

    #x_train[np.isnan(x_train)] = -100
    gc.collect()
    logger.info("data size {}".format(x_train.shape))
    logger.info('load end')
    all_params = {'max_depth': [5],
                  'learning_rate': [0.1],  # [0.06, 0.1, 0.2],
                  #'n_estimators': [10000],
                  'min_data_in_leaf': [10],
                  'feature_fraction': [0.9],
                  #'boosting_type': ['dart'],  # ['gbdt'],
                  #'xgboost_dart_mode': [False],
                  #'num_leaves': [96],
                  'metric': ['binary_logloss'],
                  'bagging_fraction': [0.9],
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

    min_score = (100, 100, 100)
    min_params = None
    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0

    for params in tqdm(list(ParameterGrid(all_params))):

        cnt = 0
        list_score = []
        list_score2 = []
        list_score3 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv:

            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test]

            list_idx = df.loc[test].reset_index(drop=True).groupby(
                'order_id').apply(lambda x: x.index.values.shape[0]).tolist()
            list_idx = np.array(list_idx, dtype=np.int)

            train_data = lgb.Dataset(trn_x, label=trn_y)
            test_data = lgb.Dataset(val_x, label=val_y)
            clf = lgb.train(params,
                            train_data,
                            3000,  # params['n_estimators'],
                            # early_stopping_rounds=30,
                            valid_sets=[test_data],
                            feval=f1_metric,
                            fobj=logregobj)
            pred = clf.predict(val_x)
            all_pred[test] = pred

            _score = log_loss(val_y, pred)
            _score2 = - roc_auc_score(val_y, pred)
            _score3 = - f1_score(val_y, pred > THRESH)
            logger.info('   _score: %s' % _score)
            list_score.append(_score)
            list_score2.append(_score2)
            list_score3.append(_score3)
            if clf.best_iteration != -1:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])

            with open('train_cv_pred.pkl', 'wb') as f:
                pickle.dump(pred, f, -1)
            del trn_x
            del clf
            gc.collect()
            # break

        with open('train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        trees = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))
        score3 = (np.mean(list_score3), np.min(list_score3), np.max(list_score3))

        logger.info('param: %s' % (params))
        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('score: {} (avg min max {})'.format(score2[use_score], score2))
        logger.info('score3: {} (avg min max {})'.format(score3[use_score], score2))
        if min_score[use_score] > score[use_score]:
            min_score = score
            min_score2 = score2
            min_score3 = score3
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))
        logger.info('best score2: {} {}'.format(min_score2[use_score], min_score2))
        logger.info('best score3: {} {}'.format(min_score3[use_score], min_score3))
        logger.info('best_param: {}'.format(min_params))

    gc.collect()
    train_data = lgb.Dataset(x_train, label=y_train)
    logger.info('train start')
    clf = lgb.train(min_params,
                    train_data,
                    trees,
                    valid_sets=[train_data])
    logger.info('train end')
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    ###
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    n_features = imp.shape[0]
    imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
    logger.info('imp use {} {}'.format(imp_use.shape, n_features))
    with open('features_train.py', 'w') as f:
        f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

    with open('fillna_mean.pkl', 'rb') as f:
        fillna_mean = pickle.load(f)

    x_test = load_test_data()
    #x_test.drop(DROP_FEATURE, axis=1, inplace=True)
    with open('0705_old_rate001/test_tmp.pkl', 'rb') as f:
        x_test['first'] = pickle.load(f)[:, 1]

    x_test = x_test.fillna(fillna_mean).values

    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))
    logger.info('train end')
    _p_test = clf.predict(x_test)
    p_test = np.zeros((_p_test.shape[0], 2))
    p_test[:, 1] = _p_test
    with open('test_tmp.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)
