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

import tensorflow as tf
from tqdm import tqdm
from tffm import TFFMClassifier
from features_drop import DROP_FEATURE
now_order_ids = None
THRESH = 0.189

list_idx = None
cnt_a = 0


def aaa(arg):
    return f1_score(*arg)


from utils import f1, f1_group#, f1_group_idx

DIR = 'result_tmp_fm/'


def f1_metric(label, pred):
    res = f1_group(label, pred, list_idx)
    sc = np.mean(res)
    logger.debug('f1: %s' % (sc))
    return 'f1', sc, True

COLUMN_NAMES = ["order_id", "product_id", "user_id", "aisle_id", "department_id", "reordered"]
def get_dict(ids):
    map_idx2user = dict([(i, ids[i]) for i in range(len(ids))])
    map_user2idx = dict([(ids[i], i) for i in range(len(ids))])
    return map_idx2user, map_user2idx

import scipy.sparse as spMat
import glob
TRAIN_DATA_FOLDER = '../data/dmt_train_only_rebuy/'
TEST_DATA_FOLDER = '../data/dmt_test_only_rebuy/'

def rrr():
    df1 = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id', 'reordered'])
    return pd.merge(df1, df, how='left', on=['order_id', 'user_id', 'product_id'])['reordered'].fillna(0).values

def make_data():
    path = '../input/df_prior.csv'
    data = pd.read_csv(path,
                     usecols=COLUMN_NAMES,
                     dtype=np.int)
    a = data['reordered'].values
    data['score'] = np.where(a == 1, 1., 0.).astype(np.float32) #numpy.ones(data.shape[0], dtype=numpy.int8)

    #order_ids = np.sort(data["order_id"].unique())
    item_ids = np.sort(data["product_id"].unique())
    user_ids = np.sort(data["user_id"].unique())
    aisle_ids = np.sort(data["aisle_id"].unique())
    depart_ids = np.sort(data["department_id"].unique())

    map_idx2user, map_user2idx = get_dict(user_ids)
    map_idx2item, map_item2idx = get_dict(item_ids)
    map_idx2aisle, map_aisle2idx = get_dict(aisle_ids)
    map_idx2depart, map_depart2idx = get_dict(depart_ids)
    with open('fm_data/map_data.pkl', 'wb') as f:
        pickle.dump([map_idx2user, map_user2idx, map_idx2item, map_item2idx,
                         map_idx2aisle, map_aisle2idx,map_idx2depart, map_depart2idx], f, -1)


    df = data.groupby(['user_id', 'product_id'])['score'].sum().reset_index()
    df.columns = ['user_id', 'product_id', 'score']
    df["user_id"] = df["user_id"].apply(lambda x: map_user2idx[x])
    df["product_id"] = df["product_id"].apply(lambda x: map_item2idx[x])
    
    A = spMat.coo_matrix(
        (df["score"], (df["user_id"], df["product_id"])),
        shape=(len(user_ids), len(item_ids))
    ).tolil()
    mu = A.sum(axis=1)    
    for i, m in tqdm(enumerate(mu)):
        d = np.sum(m)
        if d != 0:
            A.data[i] = [t / d for t in A.data[i]]
    logger.info('user_item {}'.format(A.shape))
    with open('fm_data/user_item.pkl', 'wb') as f:
        pickle.dump(A, f, -1)

    A = spMat.coo_matrix(
        (df["score"], (df["user_id"], df["product_id"])),
        shape=(len(user_ids), len(item_ids))
    ).T.tolil()
    mu = A.sum(axis=1)    
    for i, m in tqdm(enumerate(mu)):
        d = np.sum(m)
        if d != 0:
            A.data[i] = [t / d for t in A.data[i]]
    logger.info('item_user {}'.format(A.shape))            
    with open('fm_data/item_user.pkl', 'wb') as f:
        pickle.dump(A, f, -1)


    df = data.groupby(['user_id', 'aisle_id'])['score'].sum().reset_index()
    df.columns = ['user_id', 'aisle_id', 'score']
    df["user_id"] = df["user_id"].apply(lambda x: map_user2idx[x])
    df["aisle_id"] = df["aisle_id"].apply(lambda x: map_aisle2idx[x])
    
    A = spMat.coo_matrix(
        (df["score"], (df["user_id"], df["aisle_id"])),
        shape=(len(user_ids), len(aisle_ids))
    ).tolil()
    mu = A.sum(axis=1)    
    for i, m in tqdm(enumerate(mu)):
        d = np.sum(m)
        if d != 0:
            A.data[i] = [t / d for t in A.data[i]]
    logger.info('user_aisle {}'.format(A.shape))            
    with open('fm_data/user_aisle.pkl', 'wb') as f:
        pickle.dump(A, f, -1)

    df = data.groupby(['user_id', 'department_id'])['score'].sum().reset_index()
    df.columns = ['user_id', 'department_id', 'score']
    df["user_id"] = df["user_id"].apply(lambda x: map_user2idx[x])
    df["department_id"] = df["department_id"].apply(lambda x: map_aisle2idx[x])
    
    A = spMat.coo_matrix(
        (df["score"], (df["user_id"], df["department_id"])),
        shape=(len(user_ids), len(aisle_ids))
    ).tolil()
    mu = A.sum(axis=1)    
    for i, m in tqdm(enumerate(mu)):
        d = np.sum(m)
        if d != 0:
            A.data[i] = [t / d for t in A.data[i]]
    logger.info('user_depart {}'.format(A.shape))            
    with open('fm_data/user_depart.pkl', 'wb') as f:
        pickle.dump(A, f, -1)
        

def load_train_data():
    logger.info('load data')
    df = read_multi_csv(TRAIN_DATA_FOLDER)
    df = df.sort_values(['order_id', 'user_id', 'product_id']).reset_index(drop=True)

    logger.info('load abase data')
    with open('fm_data/map_data.pkl', 'rb') as f:
        map_idx2user, map_user2idx, map_idx2item, map_item2idx, map_idx2aisle, map_aisle2idx,map_idx2depart, map_depart2idx = pickle.load(f)

    df["user_id"] = df["user_id"].apply(lambda x: map_user2idx[x])
    df["product_id"] = df["product_id"].apply(lambda x: map_item2idx[x])
    df["aisle_id"] = df["aisle_id"].apply(lambda x: map_aisle2idx[x])
    df["department_id"] = df["department_id"].apply(lambda x: map_depart2idx[x])
    df['score'] = np.ones(df.shape[0])
    
    A_item = spMat.coo_matrix(
        (df["score"], (df.index.values, df["product_id"])),
        shape=(df.shape[0], len(map_idx2item))
    ).tocsr()
    A_user = spMat.coo_matrix(
        (df["score"], (df.index.values, df["user_id"])),
        shape=(df.shape[0], len(map_idx2user))
    ).tocsr()
    A_aisle = spMat.coo_matrix(
        (df["score"], (df.index.values, df["aisle_id"])),
        shape=(df.shape[0], len(map_idx2aisle))
    ).tocsr()
    A_depart = spMat.coo_matrix(
        (df["score"], (df.index.values, df["department_id"])),
        shape=(df.shape[0], len(map_idx2depart))
    ).tocsr()
    logger.info('load A data')
    A = spMat.hstack([A_user, A_item, A_aisle, A_depart])

    del A_user
    del A_item
    del A_aisle
    del A_depart
    gc.collect()
    logger.info('stack A data')
    with open('fm_data/user_item.pkl', 'rb') as f:
        A1 = pickle.load(f).tocsr()[df["user_id"].values, :]
    gc.collect()
    """
    logger.info('stack A1 data')    
    with open('fm_data/item_user.pkl', 'rb') as f:
        A2 = pickle.load(f).tocsr()[df["product_id"].values, :]
    gc.collect()
    """
    logger.info('stack A2 data')        
    with open('fm_data/user_aisle.pkl', 'rb') as f:
        A3 = pickle.load(f).tocsr()[df["user_id"].values, :]
    gc.collect()
    logger.info('stack A3 data')        
    with open('fm_data/user_depart.pkl', 'rb') as f:
        A4 = pickle.load(f).tocsr()[df["user_id"].values, :]
    logger.info('load As data')
    #A = spMat.hstack([A, A1, A2, A3, A4])
    A = spMat.hstack([A, A1, A3, A4])
    logger.info('stack As data')
    with open('user_split.pkl', 'rb') as f:
        cv = pickle.load(f)

    list_cv = []
    user_ids = df['user_id']
    for train, test in cv:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)
        list_cv.append((trn, val))
    logger.info('dump data')
    target = rrr()
    with open('fm_data/train_data.pkl', 'wb') as f:
        pickle.dump((A, target, list_cv), f, -1)
    logger.info('end')
    return A, target, list_cv

def load_test_data():
    logger.info('load data')
    df = read_multi_csv(TEST_DATA_FOLDER)
    df = df.sort_values(['o_order_id', 'o_user_id', 'o_product_id']).reset_index(drop=True)

    logger.info('load base data')
    with open('fm_data/map_data.pkl', 'rb') as f:
        map_idx2user, map_user2idx, map_idx2item, map_item2idx, map_idx2aisle, map_aisle2idx,map_idx2depart, map_depart2idx = pickle.load(f)

    df["user_id"] = df["user_id"].apply(lambda x: map_user2idx[x])
    df["product_id"] = df["product_id"].apply(lambda x: map_item2idx[x])
    df["aisle_id"] = df["aisle_id"].apply(lambda x: map_aisle2idx[x])
    df["department_id"] = df["department_id"].apply(lambda x: map_aisle2idx[x])
    df['score'] = np.ones(df.shape[0])
    
    A_item = spMat.coo_matrix(
        (df["score"], (df.index.values, df["product_id"])),
        shape=(df.shape[0], len(map_idx2item))
    ).tocsr()
    A_user = spMat.coo_matrix(
        (df["score"], (df.index.values, df["user_id"])),
        shape=(df.shape[0], len(map_idx2user))
    ).tocsr()
    A_aisle = spMat.coo_matrix(
        (df["score"], (df.index.values, df["aisle_id"])),
        shape=(df.shape[0], len(map_idx2aisle))
    ).tocsr()
    A_depart = spMat.coo_matrix(
        (df["score"], (df.index.values, df["department_id"])),
        shape=(df.shape[0], len(map_idx2depart))
    ).tocsr()

    A = spMat.hstack([A_user, A_item, A_aisle, A_depart])
    del A_user
    del A_item
    del A_aisle
    del A_depart
    gc.collect()
    with open('fm_data/user_item.pkl', 'rb') as f:
        A1 = pickle.load(f).tocsr()[df["user_id"].values, :]
    gc.collect()
    """
    with open('fm_data/item_user.pkl', 'rb') as f:
        A2 = pickle.load(f).tocsr()[df["product_id"].values, :]
    gc.collect()
    """
    with open('fm_data/user_aisle.pkl', 'rb') as f:
        A3 = pickle.load(f).tocsr()[df["user_id"].values, :]
    gc.collect()
    with open('fm_data/user_depart.pkl', 'rb') as f:
        A4 = pickle.load(f).tocsr()[df["user_id"].values, :]

    A = spMat.hstack([A, A1, A3, A4])

    with open('fm_data/test_data.pkl', 'wb') as f:
        pickle.dump(A, f, -1)
    
    return A

def read_csv(filename):
    logger.info(filename)
    df = pd.read_csv(filename, usecols=['o_order_id', 'o_user_id', 'o_product_id', 'p_aisle_id', 'p_department_id'], dtype=np.int)
    df = df[['o_order_id', 'o_user_id', 'o_product_id', 'p_aisle_id', 'p_department_id']]
    df.columns = ["order_id", "user_id", "product_id", "aisle_id", "department_id"]
    return df

def read_multi_csv(folder):
    logger.info('enter')
    paths = glob.glob(folder + '/*.csv.gz')
    logger.info(folder)
    logger.info('file_num: %s' % len(paths))
    df = None  # pd.DataFrame()
    p = Pool()
    df = pd.concat(p.map(read_csv, paths), ignore_index=True, copy=False)
    p.close()
    p.join()
    logger.info('exit')
    return df
    

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

    logger.info('load start')
    #make_data()

    #x_train, y_train, cv = load_train_data()
    with open('fm_data/train_data.pkl', 'rb') as f:
        x_train, y_train, cv = pickle.load(f)
    x_train = x_train.tocsr()
    df = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)


    min_score = (100, 100, 100)
    min_params = None
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0
    logger.info('load end {}'.format(x_train.shape))
    #for params in tqdm(list(ParameterGrid(all_params))):

    gc.collect()
    if 1:
        cnt = -1
        list_score = []
        list_score2 = []
        list_score3 = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv:
            cnt += 1
            idx = np.arange(x_train.shape[0], dtype=int)
            _train = idx[train]
            _test = idx[test]            
            trn_x = x_train[_train, :]
            val_x = x_train[_test, :]
            trn_y = y_train[train]
            val_y = y_train[test]

            list_idx = df.loc[test].reset_index(drop=True).groupby(
                'order_id').apply(lambda x: x.index.values.shape[0]).tolist()
            list_idx = np.array(list_idx, dtype=np.int)

            clf = TFFMClassifier(order=2,
                                 rank=10,
                                     optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                     n_epochs=100,
                                     batch_size=100000,
                                     init_std=0.001,
                                     input_type='sparse'
                                     )
            
            clf.fit(trn_x, trn_y, show_progress=True)
            pred = clf.predict_proba(val_x)[:, 1]
            all_pred[test] = pred

            _score = log_loss(val_y, pred)
            _score2 = - roc_auc_score(val_y, pred)
            _, _score3, _ = f1_metric(val_y.astype(int), pred.astype(float))
            logger.debug('   _score: %s' % _score3)
            list_score.append(_score)
            list_score2.append(_score2)
            list_score3.append(- 1 * _score3)

            with open(DIR + 'train_cv_pred_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(pred, f, -1)
            """
            with open(DIR + 'model_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(clf, f, -1)
            """
            del trn_x
            del clf
            gc.collect()
            #break
        with open(DIR + 'train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))
        score3 = (np.mean(list_score3), np.min(list_score3), np.max(list_score3))

        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('score: {} (avg min max {})'.format(score2[use_score], score2))
        logger.info('score3: {} (avg min max {})'.format(score3[use_score], score2))
        if min_score[use_score] > score3[use_score]:
            min_score = score3
            min_score2 = score2
            min_score3 = score3
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))
        logger.info('best score2: {} {}'.format(min_score2[use_score], min_score2))
        logger.info('best score3: {} {}'.format(min_score3[use_score], min_score3))
        logger.info('best_param: {}'.format(min_params))

    gc.collect()

    # for params in tqdm(list(ParameterGrid(all_params))):
    #    min_params = params
    clf = TFFMClassifier(order=2,
                                 rank=10,
                                     optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                     n_epochs=50,
                                     batch_size=100000,
                                     init_std=0.001,
                                     input_type='sparse'
                                     )
            
    clf.fit(x_train, y_train,show_progress=True)

    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    ###
    with open(DIR + 'model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open(DIR + 'usecols.pkl', 'rb') as f:
        usecols = pickle.load(f)

    x_test = load_test_data()

    gc.collect()

    logger.info('train end')
    p_test = clf.predict_proba(x_test)
    with open(DIR + 'test_tmp.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)
