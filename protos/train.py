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

list_idx = None
cnt_a = 0


def aaa(arg):
    return f1_score(*arg)


from utils import f1, f1_group


def f1_metric(label, pred):
    #res = [f1(label.take(i), pred.take(i)) for i in list_idx]
    res = f1_group(label, pred, list_idx)
    sc = np.mean(res)
    logger.debug('f1: %s' % (sc))
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
    """
    all_params = {'max_depth': [7],
                  'learning_rate': [0.1],  # [0.06, 0.1, 0.2],
                  'n_estimators': [1000],
                  'min_data_in_leaf': [10],
                  'feature_fraction': [0.8],
                  #'boosting_type': ['rf'],  # ['gbdt'],
                  #'xgboost_dart_mode': [False],
                  'num_leaves': [255],
                  'bagging_fraction': [0.9],
                  #'min_child_samples': [10],
                  'lambda_l1': [1, 0],
                  #'lambda_l2 ': [1, 0],
                  #'max_bin': [127],
                  #'min_sum_hessian_in_leaf': [0, 0.01],
                  #'min_gain_to_split': [0, 0.1, 0.01],
                  #'min_data_in_bin': [1, 3, 5, 10],
                  'silent': [True],
                  'seed': [114514]
                  }
    """
    all_params = {'min_child_weight': [10],
                  'subsample': [0.9],
                  'seed': [114514],
                  'n_estimators': [1000, 2000, 3000, 4000, 5000],
                  'colsample_bytree': [0.9],
                  'silent': [False],
                  'learning_rate': [0.1],
                  'max_depth': [5],
                  'min_data_in_bin': [3],
                  'min_split_gain': [0],
                  'reg_alpha': [1],
                  'max_bin': [511],
                  #'objective': ['xentropy'],
                  #'metric_freq': [100]
                  }
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
            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]

            val_y = y_train[test]

            trn_w = sample_weight[train]
            val_w = sample_weight[test]

            list_idx = df.loc[test].reset_index(drop=True).groupby(
                'order_id').apply(lambda x: x.index.values.shape[0]).tolist()
            list_idx = np.array(list_idx, dtype=np.int)

            clf = LGBMClassifier(**params)
            clf.fit(trn_x, trn_y,
                    # sample_weight=trn_w,
                    # eval_sample_weight=[val_w],
                    #eval_set=[(val_x, val_y)],
                    verbose=True,
                    # eval_metric=f1_metric,
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

            with open('train_cv_pred_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(pred, f, -1)
            with open('model_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(clf, f, -1)
            del trn_x
            del clf
            gc.collect()
            break
        with open('train_cv_tmp.pkl', 'wb') as f:
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

    clf = LGBMClassifier(**min_params)
    clf.fit(x_train, y_train,
            verbose=True,
            eval_metric="auc",
            # sample_weight=sample_weight
            )
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    ###
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('usecols.pkl', 'rb') as f:
        usecols = pickle.load(f)

    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv('feature_importances.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    with open('fillna_mean.pkl', 'rb') as f:
        fillna_mean = pickle.load(f)

    x_test = load_test_data()
    logger.info('product_last')
    x_test = x_test.merge(pd.read_csv('product_last.csv').astype(np.float32), how='left',
                          left_on='o_product_id', right_on='product_id', copy=False)
    logger.info('product_first')
    x_test = x_test.merge(pd.read_csv('product_first.csv').astype(np.float32), how='left',
                          left_on='o_product_id', right_on='product_id', copy=False)
    logger.info('product_all')
    x_test = x_test.merge(pd.read_csv('product_all.csv').astype(np.float32), how='left',
                          left_on='o_product_id', right_on='product_id', copy=False)
    logger.info('product_wordpred')
    x_test = x_test.merge(pd.read_csv('word_preds.csv').astype(np.float32), how='left',
                          left_on='o_product_id', right_on='product_id', copy=False)
    x_test = x_test.merge(pd.read_csv('user_item_pattern.csv').astype(np.float32).rename(columns={'user_id': 'o_user_id'}), how='left',
                          on='o_user_id', copy=True)

    id_cols = [col for col in x_test.columns.values
               if re.search('_id$', col) is not None and
               col not in set(['o_user_id', 'o_product_id', 'p_aisle_id', 'p_department_id'])]
    logger.debug('id_cols {}'.format(id_cols))
    x_test.drop(id_cols, axis=1, inplace=True)
    logger.info('usecols')
    # x_test['0714_10000loop'] = get_stack('0714_10000loop/', is_train=False)
    # x_test['0715_2nd_order'] = get_stack('0715_2nd_order/', is_train=False)

    # x_test.drop(usecols, axis=1, inplace=True)
    # x_test = x_test[FEATURE]

    x_test = x_test[usecols]
    gc.collect()
    logger.info('values {} {}'.format(len(usecols), x_test.shape))
    x_test.fillna(fillna_mean, inplace=True)
    logger.info('fillna')
    # x_test = x_test.as_matrix()
    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))
    logger.info('train end')
    p_test = clf.predict_proba(x_test)
    with open('test_tmp.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)
