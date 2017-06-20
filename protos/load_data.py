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


def read_csv(filename):
    df = pd.read_csv(filename).astype(np.float32)
    return df


def read_multi_csv(folder):
    paths = glob.glob(folder + '/*.csv.gz')
    logger.info(folder)
    logger.info('file_num: %s' % len(paths))
    df = None  # pd.DataFrame()
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
    return df


def load_train_data():
    logger.info('enter')

    if os.path.exists(TRAIN_DATA_PATH):
        logger.info('using seriarized data')
        with open(TRAIN_DATA_PATH, 'rb') as f:
            return pickle.load(f)

    logger.info('load data')
    #df = read_multi_csv(TRAIN_DATA_FOLDER)
    # with open('train_data.pkl', 'wb') as f:
    #    pickle.dump(df, f, -1)
    with open('train_data.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('train_baseline.pkl', 'rb') as f:
        df2, _ = pickle.load(f)
    f_to_use = ['order_id', 'user_id', 'product_id',
                'user_total_orders', 'user_total_items', 'total_distinct_items',
                'user_average_days_between_orders', 'user_average_basket',
                'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
                'UP_delta_hour_vs_last']  # 'dow', 'UP_same_dow_as_last_order'
    df2 = df2[f_to_use]
    df = pd.merge(df, df2, how='left', left_on=['o_order_id', 'o_user_id', 'o_product_id'],
                  right_on=['order_id', 'user_id', 'product_id'])

    _df = df[['o_order_id', 'o_user_id', 'o_product_id']]
    _df.columns = ['order_id', 'user_id', 'product_id']
    _df.to_csv('train_data_idx.csv', index=False)

    with open('user_split.pkl', 'rb') as f:
        cv = pickle.load(f)

    list_cv = []
    user_ids = df['o_user_id']
    for train, test in cv:
        trn = user_ids.isin(train)
        val = user_ids.isin(test)
        list_cv.append((trn, val))

    logger.info('etl data')
    target = df['target'].values
    df.drop('target', axis=1, inplace=True)

    id_cols = [col for col in df.columns.values if re.search('_id$', col) is not None]
    df.drop(id_cols, axis=1, inplace=True)

    cum_cols = [col for col in df.columns.values if re.search('cum', col) is not None]
    df['since_last_order'] = df['o_cum_days'] - df['l_cum_days']
    df['since_last_aisle'] = df['o_cum_days'] - df['la_cum_days']
    df['since_last_depart'] = df['o_cum_days'] - df['ld_cum_days']
    #df.drop(cum_cols, axis=1, inplace=True)

    def normarize(col_name):
        cols = [col for col in df.columns.values if re.search(col_name, col) is not None]
        row_sum = df[cols].sum(axis=1)
        for col in cols:
            df[col] /= row_sum

    cols = [col for col in df.columns.values if re.search('u3_order_hour_of_day', col) is not None]
    row_sum = df[cols].sum(axis=1)
    for col in cols:
        df[col] /= row_sum

    normarize('u2_order_dow')
    normarize('u3_order_hour_of_day')
    normarize('u4_department_id')

    normarize('i2_order_dow')
    normarize('i3_order_hour_of_day')
    normarize('i4_department_id')

    normarize('ui_order_dow')

    logger.info('dump data')
    with open(TRAIN_DATA_PATH, 'wb') as f:
        pickle.dump((df, target, list_cv), f, -1)
    logger.info('exit')
    return df, target, list_cv


def load_test_data():
    logger.info('enter')

    if os.path.exists(TEST_DATA_PATH):
        logger.info('using seriarized data')
        with open(TEST_DATA_PATH, 'rb') as f:
            return pickle.load(f)

    logger.info('load data')
    #df = read_multi_csv(TEST_DATA_FOLDER)
    # with open('test_data.pkl', 'wb') as f:
    #    pickle.dump(df, f, -1)
    with open('test_data.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('test_baseline.pkl', 'rb') as f:
        df2 = pickle.load(f)
    f_to_use = ['order_id', 'user_id', 'product_id',
                'user_total_orders', 'user_total_items', 'total_distinct_items',
                'user_average_days_between_orders', 'user_average_basket',
                'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
                'UP_delta_hour_vs_last']  # 'dow', 'UP_same_dow_as_last_order'
    df2 = df2[f_to_use]
    df = pd.merge(df, df2, how='left', left_on=['o_order_id', 'o_user_id', 'o_product_id'],
                  right_on=['order_id', 'user_id', 'product_id'])

    _df = df[['o_order_id', 'o_user_id', 'o_product_id']]
    _df.columns = ['order_id', 'user_id', 'product_id']
    _df.to_csv('test_data_idx.csv', index=False)

    logger.info('etl data')
    id_cols = [col for col in df.columns.values if re.search('_id$', col) is not None]
    df.drop(id_cols, axis=1, inplace=True)

    cum_cols = [col for col in df.columns.values if re.search('cum', col) is not None]
    df['since_last_order'] = df['o_cum_days'] - df['l_cum_days']
    df['since_last_aisle'] = df['o_cum_days'] - df['la_cum_days']
    df['since_last_depart'] = df['o_cum_days'] - df['ld_cum_days']
    #df.drop(cum_cols, axis=1, inplace=True)

    def normarize(col_name):
        cols = [col for col in df.columns.values if re.search(col_name, col) is not None]
        row_sum = df[cols].sum(axis=1)
        for col in cols:
            df[col] /= row_sum

    cols = [col for col in df.columns.values if re.search('u3_order_hour_of_day', col) is not None]
    row_sum = df[cols].sum(axis=1)
    for col in cols:
        df[col] /= row_sum

    normarize('u2_order_dow')
    normarize('u3_order_hour_of_day')
    normarize('u4_department_id')

    normarize('i2_order_dow')
    normarize('i3_order_hour_of_day')
    normarize('i4_department_id')

    normarize('ui_order_dow')

    logger.info('dump data')
    with open(TEST_DATA_PATH, 'wb') as f:
        pickle.dump(df, f, -1)

    logger.info('exit')
    return df


if __name__ == '__main__':

    df = read_multi_csv(TRAIN_DATA_FOLDER)
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(df, f, -1)
    df = read_multi_csv(TEST_DATA_FOLDER)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(df, f, -1)
