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

from tqdm import tqdm

from load_data import load_train_data, load_test_data

now_order_ids = None
THRESH = 0.194

val_data = None
cv_idx = None


def f1_metric(label, pred):
    pred = pred > THRESH

    df = val_data[cv_idx].copy()
    df['label'] = label
    df['pred_label'] = pred

    sc = f1_score(df.label.values.astype(np.bool), df.pred_label.values.astype(np.bool))
    return 'f1', sc, True


def _f1_metric(label, pred):
    return 'auc', roc_auc_score(label, pred), True
    """
    df = pd.DataFrame({'order_id': now_order_ids, 'label': label, 'pred': pred > THRESH})
    scores = []
    for order_id in df.order_id.unique():
        tmp = df[df['order_id'] == order_id]
        sc = f1_score(tmp['label'].values, tmp['pred'].values)
        scores.append(sc)
    scores = df.groupby('order_id').apply(lambda tmp: f1_score(tmp['label'], tmp['pred']))
    return 'f1', np.mean(scores), True
    """

if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler('stack.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('load start')
    df = pd.read_csv('train_data_idx.csv')
    with open('train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)

    df1 = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id', 'reordered'])
    target = pd.merge(df, df1, how='left', on=['order_id', 'user_id', 'product_id'])['reordered'].fillna(0).values
    df.target = target
        
        
    df['pred'] = pred
    df_order = df.groupby('order_id')['pred'].agg({'o_avg': np.mean, 'o_min': np.min, 'o_max': np.max, 'o_cnt': len}).reset_index()
    df = pd.merge(df, df_order, how='left', on='order_id')
    df['rank'] = df.groupby('order_id')['pred'].rank(ascending=False)

    #print(df.sort_values('order_id')[['order_id', 'rank', 'pred']].head(100))

    with open('user_split.pkl', 'rb') as f:
        cv = pickle.load(f)

    id_cols = [col for col in df.columns.values if re.search('_id$', col) is not None] + ['target']
    val_data = df
    x_train = df.drop(id_cols, axis=1)
    train_cols = x_train.columns.values

    x_train = x_train.values
    y_train = df.target.values
    sample_weight = 1 / df['o_cnt'].values

    list_cv = []
    user_ids = df['user_id']

    for train, test in cv:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)
        list_cv.append((trn, val))

    logger.info('load end')
    all_params = {'max_depth': [3],
                  'learning_rate': [0.1],  # [0.06, 0.1, 0.2],
                  'n_estimators': [130],#[10000],
                  'min_child_weight': [30],
                  'colsample_bytree': [0.9],
                  #'boosting_type': ['dart'],  # ['gbdt'],
                  #'xgboost_dart_mode': [False],
                  #'num_leaves': [96],
                  'subsample': [0.9],
                  #'min_child_samples': [10],
                  #'reg_alpha': [1],
                  #'reg_lambda': [1],
                  #'max_bin': [500],
                  'min_split_gain': [0],
                  'silent': [True],
                  'seed': [6436]
                  }

    min_score = (100, 100, 100)
    min_params = None
    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0
    """
    for params in tqdm(list(ParameterGrid(all_params))):

        cnt = 0
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in list_cv:
            cv_idx = test

            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test]

            trn_w = sample_weight[train]
            val_w = sample_weight[test]

            clf = LGBMClassifier(**params)
            clf.fit(trn_x, trn_y,
                    # sample_weight=trn_w,
                    # eval_sample_weight=[val_w],
                    eval_set=[(val_x, val_y)],
                    verbose=True,
                    eval_metric=f1_metric,
                    early_stopping_rounds=30
                    )
            pred = clf.predict_proba(val_x)[:, 1]
            all_pred[test] = pred

            _score = log_loss(val_y, pred)
            _score2 = - roc_auc_score(val_y, pred)
            # logger.debug('   _score: %s' % _score)
            list_score.append(_score)
            list_score2.append(_score2)
            if clf.best_iteration != -1:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])

            with open('train_cv_pred_2.pkl', 'wb') as f:
                pickle.dump(pred, f, -1)

        with open('train_cv_tmp_2.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        params['n_estimators'] = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('score: {} (avg min max {})'.format(score2[use_score], score2))
        if min_score[use_score] > score[use_score]:
            min_score = score
            min_score2 = score2
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))
        logger.info('best score2: {} {}'.format(min_score2[use_score], min_score2))
        logger.info('best_param: {}'.format(min_params))

    gc.collect()
    """
    for params in tqdm(list(ParameterGrid(all_params))):
        min_params = params
    clf = LGBMClassifier(**min_params)
    clf.fit(x_train, y_train,
            #sample_weight=sample_weight
            )
    with open('model2.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    ###
    with open('model2.pkl', 'rb') as f:
        clf = pickle.load(f)
    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    n_features = imp.shape[0]
    imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
    logger.info('imp use {} {}'.format(imp_use.shape, n_features))
    with open('features_train2.py', 'w') as f:
        f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

    df = pd.read_csv('test_data_idx.csv')
    with open('test_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[:, 1]

    df['pred'] = pred
    df_order = df.groupby('order_id')['pred'].agg(
        {'o_avg': np.mean, 'o_min': np.min, 'o_max': np.max, 'o_cnt': len}).reset_index()
    df = pd.merge(df, df_order, how='left', on='order_id')
    df['rank'] = df.groupby('order_id')['pred'].rank(ascending=False)

    x_test = df[train_cols].values
    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))
    logger.info('train end')
    p_test = clf.predict_proba(x_test)
    with open('test_tmp2.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)
