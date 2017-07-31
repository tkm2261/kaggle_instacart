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
    df = pd.read_csv(filename).astype(np.float32)

    def normarize(col_name):
        cols = [col for col in df.columns.values if re.search(col_name, col) is not None]
        logger.info('{} {}'.format(col_name, len(cols)))
        row_sum = df.loc[:, cols].sum(axis=1)
        for col in cols:
            df[col] /= row_sum

    def normarize2(col_name):
        cols = [col for col in df.columns.values if re.search(col_name, col) is not None]
        #logger.info('{} {}'.format(col_name, cols))
        row_sum = df.loc[:, cols].sum(axis=1)
        for col in cols:
            df[col + '_rate'] = df[col] / row_sum

    def drop(col_name):
        cols = [col for col in df.columns.values if re.search(col_name, col) is not None]
        logger.debug('{} {}'.format(col_name, cols))
        df.drop(cols, axis=1, inplace=True)

    normarize('u_u2_order_dow')
    normarize('u_u3_order_hour_of_day')
    normarize('u_u4_department_id')
    gc.collect()
    normarize('i_i2_order_dow')
    normarize('i_i3_order_hour_of_day')
    normarize('i_i4_department_id')

    drop('^u2_')
    gc.collect()
    drop('^i2_')

    normarize('ui_order_dow')
    normarize('u3_order_hour_of_day')
    gc.collect()

    # drop('^da')
    # drop('^dd')
    #cum_cols = [col for col in df.columns.values if re.search('cum', col) is not None]
    df['since_last_order'] = (df['o_cum_days'] - df['l_cum_days']).astype(np.float32)
    df['since_last_aisle'] = (df['o_cum_days'] - df['la_cum_days']).astype(np.float32)
    df['since_last_depart'] = (df['o_cum_days'] - df['ld_cum_days']).astype(np.float32)

    df['since_last_order2'] = (df['o_cum_days'] - df['l2_cum_days']).astype(np.float32)
    df['since_last_aisle2'] = (df['o_cum_days'] - df['la2_cum_days']).astype(np.float32)
    df['since_last_depart2'] = (df['o_cum_days'] - df['ld2_cum_days']).astype(np.float32)

    df['since_last_order_diff'] = (df['since_last_order'] - df['since_last_order2']).astype(np.float32)
    df['since_last_aisle_diff'] = (df['since_last_aisle'] - df['since_last_aisle2']).astype(np.float32)
    df['since_last_depart_diff'] = (df['since_last_depart'] - df['since_last_depart2']).astype(np.float32)

    df['since_last_visit_order'] = (df['o_order_number'] - df['l_max_order_number']).astype(np.float32)
    df['since_last_visit_aisle'] = (df['o_order_number'] - df['la_max_order_number']).astype(np.float32)
    df['since_last_visit_depart'] = (df['o_order_number'] - df['ld_max_order_number']).astype(np.float32)

    df['since_last_visit_order2'] = (df['o_order_number'] - df['l2_max_order_number']).astype(np.float32)
    df['since_last_visit_aisle2'] = (df['o_order_number'] - df['la2_max_order_number']).astype(np.float32)
    df['since_last_visit_depart2'] = (df['o_order_number'] - df['ld2_max_order_number']).astype(np.float32)
    ###
    """
    df['user_avg_days_rate'] = (df['udd_avg_diffs'] / df['du_avg_diffs']).astype(np.float32)
    df['user_avg_days_rate_30'] = (df['udd3_avg_diffs'] / df['du3_avg_diffs']).astype(np.float32)

    df['user_days_rate'] = (df['since_last_order'] / df['udd_avg_diffs']).astype(np.float32)
    df['user_days_rate_30'] = (df['since_last_order'] / df['udd3_avg_diffs']).astype(np.float32)

    df['item_days_rate'] = (df['since_last_order'] / df['di_avg_diffs']).astype(np.float32)
    df['item_days_rate_30'] = (df['since_last_order'] / df['di3_avg_diffs']).astype(np.float32)

    ###

    df['user_avg_days_diff'] = (df['udd_avg_diffs'] - df['du_avg_diffs']).astype(np.float32)
    df['user_avg_days_diff_30'] = (df['udd3_avg_diffs'] - df['du3_avg_diffs']).astype(np.float32)

    df['user_days_diff'] = (df['since_last_order'] - df['udd_avg_diffs']).astype(np.float32)
    df['user_days_diff_30'] = (df['since_last_order'] - df['udd3_avg_diffs']).astype(np.float32)

    df['item_days_diff'] = (df['since_last_order'] - df['di_avg_diffs']).astype(np.float32)
    df['item_days_diff_30'] = (df['since_last_order'] - df['di3_avg_diffs']).astype(np.float32)
    """

    ###
    df['user_item_days_rate'] = (df['since_last_order'] / df['du_avg_diffs']).astype(np.float32)
    df['user_item_days_rate_30'] = (df['since_last_order'] / df['du3_avg_diffs']).astype(np.float32)
    df['user_item_days_rate_reordered_30'] = (df['since_last_order'] / df['dui3_avg_diffs']).astype(np.float32)

    df['user_item_days_diff'] = (df['since_last_order'] - df['du_avg_diffs']).astype(np.float32)
    df['user_item_days_diff_30'] = (df['since_last_order'] - df['du3_avg_diffs']).astype(np.float32)
    df['user_item_days_diff_reordered_30'] = (df['since_last_order'] - df['dui3_avg_diffs']).astype(np.float32)

    ###
    #df['user_aisle_days_rate'] = (df['since_last_aisle'] / df['da_avg_diffs']).astype(np.float32)
    df['user_aisle_days_rate_30'] = (df['since_last_aisle'] / df['da3_avg_diffs']).astype(np.float32)
    df['user_aisle_days_rate_reordered_30'] = (df['since_last_aisle'] / df['dai3_avg_diffs']).astype(np.float32)

    #df['user_aisle_days_diff'] = (df['since_last_aisle'] - df['da_avg_diffs']).astype(np.float32)
    df['user_aisle_days_diff_30'] = (df['since_last_aisle'] - df['da3_avg_diffs']).astype(np.float32)
    df['user_aisle_days_diff_reordered_30'] = (df['since_last_aisle'] - df['dai3_avg_diffs']).astype(np.float32)

    ###
    #df['user_depart_days_rate'] = (df['since_last_depart'] / df['dd_avg_diffs']).astype(np.float32)
    df['user_depart_days_rate_30'] = (df['since_last_depart'] / df['dd3_avg_diffs']).astype(np.float32)
    df['user_depart_days_rate_reordered_30'] = (df['since_last_depart'] / df['dddi3_avg_diffs']).astype(np.float32)

    #df['user_depart_days_diff'] = (df['since_last_depart'] - df['dd_avg_diffs']).astype(np.float32)
    df['user_depart_days_diff_30'] = (df['since_last_depart'] - df['dd3_avg_diffs']).astype(np.float32)
    df['user_depart_days_diff_reordered_30'] = (df['since_last_depart'] - df['dddi3_avg_diffs']).astype(np.float32)

    id_cols = [col for col in df.columns.values
               if re.search('_id$', col) is not None and
               col not in set(['o_order_id', 'o_user_id', 'o_product_id', 'p_aisle_id', 'p_department_id'])]
    logger.debug('id_cols {}'.format(id_cols))
    df.drop(id_cols, axis=1, inplace=True)

    dropcols = sorted(list(set(df.columns.values.tolist()) & set(DROP_FEATURE)))
    df.drop(dropcols, axis=1, inplace=True)

    gc.collect()

    return df.astype(np.float32)


def read_multi_csv(folder):
    logger.info('enter')
    paths = glob.glob(folder + '/*.csv.gz')
    logger.info(folder)
    logger.info('file_num: %s' % len(paths))
    df = None  # pd.DataFrame()

    p = Pool(28)
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


def _join_data(df):
    logger.info('merges')
    df = df.merge(pd.read_csv('product_last.csv').astype(np.float32).rename(columns={'product_id': 'o_product_id'}), how='left',
                  on='o_product_id', copy=False)
    df = df.merge(pd.read_csv('product_first.csv').astype(np.float32).rename(columns={'product_id': 'o_product_id'}), how='left',
                  on='o_product_id', copy=False)
    df = df.merge(pd.read_csv('product_all.csv').astype(np.float32).rename(columns={'product_id': 'o_product_id'}), how='left',
                  on='o_product_id', copy=False)
    df = df.merge(pd.read_csv('word_preds.csv').astype(np.float32).rename(columns={'product_id': 'o_product_id'}), how='left',
                  on='o_product_id', copy=False)
    df = df.merge(pd.read_csv('user_item_pattern.csv').astype(np.float32).rename(columns={'user_id': 'o_user_id'}), how='left',
                  on='o_user_id', copy=False)
    logger.info('end')
    return df


def load_train_data():
    logger.info('enter')

    if os.path.exists(TRAIN_DATA_PATH):
        logger.info('using seriarized data')
        with open(TRAIN_DATA_PATH, 'rb') as f:
            data, target, list_cv = pickle.load(f)
            logger.info('using seriarized data2')
            target = rrr()
            return data, target, list_cv
    logger.info('load data')
    df = read_multi_csv(TRAIN_DATA_FOLDER)
    logger.info('load base data')
    with open('train_baseline.pkl', 'rb') as f:
        df2, _ = pickle.load(f)
    f_to_use = ['order_id', 'user_id', 'product_id',
                'user_total_orders', 'user_total_items', 'total_distinct_items',
                'user_average_days_between_orders', 'user_average_basket',
                'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
                'UP_delta_hour_vs_last']  # 'dow', 'UP_same_dow_as_last_order
    #f_to_use = sorted(list(set(f_to_use) & set(DROP_FEATURE)))
    df2 = df2[f_to_use].astype(np.float32)
    dropcols = sorted(list(set(df2.columns.values.tolist()) & set(DROP_FEATURE)))
    df2.drop(dropcols, axis=1, inplace=True)

    gc.collect()
    logger.info('load base merge')
    df = pd.merge(df, df2, how='left', left_on=['o_order_id', 'o_user_id', 'o_product_id'],
                  right_on=['order_id', 'user_id', 'product_id'], copy=False)

    df = df.sort_values(['o_order_id', 'o_user_id', 'o_product_id']).reset_index(drop=True)

    _df = df[['o_order_id', 'o_user_id', 'o_product_id', 'target']]
    _df.columns = ['order_id', 'user_id', 'product_id', 'target']
    _df.to_csv('train_data_idx.csv', index=False)
    del _df
    del df2
    gc.collect()

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
    gc.collect()

    logger.info('dump data')
    target = rrr()

    df = _join_data(df)
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

    df = read_multi_csv(TEST_DATA_FOLDER)
    with open('test_baseline.pkl', 'rb') as f:
        df2 = pickle.load(f)
    f_to_use = ['order_id', 'user_id', 'product_id',
                'user_total_orders', 'user_total_items', 'total_distinct_items',
                'user_average_days_between_orders', 'user_average_basket',
                'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
                'UP_delta_hour_vs_last']  # 'dow', 'UP_same_dow_as_last_order
    #f_to_use = sorted(list(set(f_to_use) & set(DROP_FEATURE)))
    df2 = df2[f_to_use].astype(np.float32)
    dropcols = sorted(list(set(df.columns.values.tolist()) & set(DROP_FEATURE)))
    df.drop(dropcols, axis=1, inplace=True)

    gc.collect()
    logger.info('load base merge')
    df = pd.merge(df, df2, how='left', left_on=['o_order_id', 'o_user_id', 'o_product_id'],
                  right_on=['order_id', 'user_id', 'product_id'], copy=False)
    df.sort_values(['o_order_id', 'o_user_id', 'o_product_id'], inplace=True)
    _df = df[['o_order_id', 'o_user_id', 'o_product_id']]
    _df.columns = ['order_id', 'user_id', 'product_id']
    _df.to_csv('test_data_idx.csv', index=False)
    del _df
    del df2
    gc.collect()

    logger.info('etl data')
    gc.collect()

    logger.info('dump data')
    df = _join_data(df)
    with open(TEST_DATA_PATH, 'wb') as f:
        pickle.dump(df, f, -1)

    logger.info('exit')
    return df


if __name__ == '__main__':
    load_train_data()
