
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

from utils import f1


def f1_metric(label, pred):
    res = [f1(label[i], pred[i]) for i in list_idx]
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


def aaa(folder):
    with open(folder + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)

    df = pd.read_csv(folder + 'train_data_idx.csv')
    df.drop('target', axis=1, inplace=True)
    df_val = df  # .loc[val, :].copy()

    df_val['pred'] = pred
    return df_val


def rrr():
    df1 = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id', 'reordered'])
    return pd.merge(df1, df, how='left', on=['order_id', 'user_id', 'product_id'])['reordered'].fillna(0)


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    all_params = {'max_depth': [5],
                  'learning_rate': [0.01],  # [0.06, 0.1, 0.2],
                  'n_estimators': [10000],
                  'min_child_weight': [10],
                  'colsample_bytree': [0.7],
                  #'boosting_type': ['rf'],  # ['gbdt'],
                  #'xgboost_dart_mode': [False],
                  #'num_leaves': [96],
                  'subsample': [0.9],
                  #'min_child_samples': [10],
                  'reg_alpha': [1],
                  #'reg_lambda': [1],
                  #'max_bin': [127],
                  'min_split_gain': [0],
                  'silent': [False],
                  'seed': [114514]
                  }

    logger.info('load start')

    df = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)

    gc.collect()

    df_val = aaa('./0715_2nd_order/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
    df_val1 = aaa('./0714_10000loop/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
    df_val2 = aaa('./0716_3rd_order_stack/').sort_values(['order_id', 'user_id', 'product_id'], ascending=False)

    y_train = rrr()
    x_train = pd.DataFrame().sort_values(['order_id', 'user_id', 'product_id'], ascending=False)
    x_train['pred1'] = df_val.pred.values
    x_train['pred2'] = df_val1.pred.values
    x_train['pred3'] = df_val2.pred.values

    x_train = x_train.values
    y_train = y_train.reordered.values

    with open('user_split.pkl', 'rb') as f:
        cv = pickle.load(f)

    cv = []
    user_ids = df_val['o_user_id']
    for train, test in cv:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)
        cv.append((trn, val))

    logger.info('load end {}'.format(x_train.shape))
    #{'seed': 6436, 'n_estimators': 809, 'learning_rate': 0.1, 'silent': True, 'subsample': 0.9, 'reg_alpha': 1, 'max_depth': 5, 'colsample_bytree': 0.7, 'min_child_weight': 5, 'max_bin': 500, 'min_split_gain': 0}
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

            list_idx = df.loc[test].reset_index(drop=True).groupby('order_id').apply(lambda x: x.index.values).tolist()
            clf = LogisticRegression(random_state=0, fit_intercept=False)
            pred = clf.predict_proba(val_x)[:, 1]
            all_pred[test] = pred

            _score = log_loss(val_y, pred)
            _score2 = - roc_auc_score(val_y, pred)
            _, _score3, _ = f1_metric(val_y.astype(int), pred.astype(float))
            logger.debug('   _score: %s' % _score3)
            list_score.append(_score)
            list_score2.append(_score2)
            list_score3.append(_score3)

            del trn_x
            del clf
            gc.collect()
            # break
        # with open('train_cv_tmp.pkl', 'wb') as f:
        #    pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
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
    """
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
    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    n_features = imp.shape[0]
    imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
    logger.info('imp use {} {}'.format(imp_use.shape, n_features))
    with open('features_train.py', 'w') as f:
        f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

    with open('fillna_mean.pkl', 'rb') as f:
        fillna_mean = pickle.load(f)

    x_test = load_test_data()

    # x_test['0714_10000loop'] = get_stack('0714_10000loop/', is_train=False)
    # x_test['0715_2nd_order'] = get_stack('0715_2nd_order/', is_train=False)

    # x_test.drop(usecols, axis=1, inplace=True)
    # x_test = x_test[FEATURE]
    x_test = x_test.fillna(fillna_mean).values

    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))
    logger.info('train end')
    p_test = clf.predict_proba(x_test, num_iteration=7000)
    with open('test_tmp.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)
    """
