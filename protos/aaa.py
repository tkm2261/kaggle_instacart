import os
import pickle
import pandas as pd
import numpy as np
import glob
import re
import gc
import os
from multiprocessing import Pool
from tqdm import tqdm
TRAIN_DATA_FOLDER = '../data/dmt_train_only_rebuy/'
TEST_DATA_FOLDER = '../data/dmt_test_only_rebuy/'

TRAIN_DATA_PATH = 'train_all.pkl'
TEST_DATA_PATH = 'test_all.pkl'

from logging import getLogger
logger = getLogger(__name__)

from features_drop import DROP_FEATURE


def read_csv(filename):
    logger.info(filename)
    df = pd.read_csv(filename, usecols=['o_order_id', 'o_user_id', 'o_product_id']).astype(np.float32)
    return df.astype(np.float32)


def read_multi_csv(folder):
    logger.info('enter')
    paths = glob.glob(folder + '/*.csv.gz')
    logger.info(folder)
    logger.info('file_num: %s' % len(paths))
    df = None  # pd.DataFrame()

    p = Pool(8)
    df = pd.concat(p.map(read_csv, paths), ignore_index=True, copy=False)
    p.close()
    p.join()
    """
    for path in tqdm(paths):
        tmp = read_csv(path)
        if df is None:
            df = tmp
            continue
        else:
            df = df.append(tmp, ignore_index=True)
        #df = pd.concat([df, tmp], ignore_index=True, copy=False)
        del tmp
        gc.collect()
    """
    logger.info('exit')
    return df


def rrr():
    df1 = pd.read_csv('train_data_idx.csv', usecols=['order_id', 'user_id', 'product_id'], dtype=int)
    df = pd.read_csv('../input/df_train.csv', usecols=['order_id', 'user_id', 'product_id', 'reordered'])
    return pd.merge(df1, df, how='left', on=['order_id', 'user_id', 'product_id'])['reordered'].fillna(0).values


def load_train_data():
    logger.info('enter')
    logger.info('load data')
    df = read_multi_csv(TRAIN_DATA_FOLDER)
    _df = df[['o_order_id', 'o_user_id', 'o_product_id', 'target']]
    _df.columns = ['order_id', 'user_id', 'product_id', 'target']
    _df.to_csv('0716_3rd_order_stack/train_data_idx.csv', index=False)


def load_test_data():
    logger.info('enter')

    logger.info('load data')

    df = read_multi_csv(TEST_DATA_FOLDER)
    _df = df[['o_order_id', 'o_user_id', 'o_product_id']]
    _df.columns = ['order_id', 'user_id', 'product_id']
    _df.to_csv('0716_3rd_order_stack/test_data_idx.csv', index=False)


if __name__ == '__main__':
    # load_train_data()
    load_test_data()
