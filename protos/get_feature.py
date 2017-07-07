
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
THRESH = 0.189

list_idx = None



if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    logger.info('load start')

    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    try:
        aaa = clf.feature_importances_
    except:
        aaa = clf.feature_importance()
        
    imp = pd.DataFrame(aaa, columns=['imp'])
    n_features = imp.shape[0]
    imp_use = imp['imp'] == 0 #.sort_values('imp', ascending=False)
    drop_col = load_test_data().columns.values#
    print(drop_col)
    drop_col = drop_col[imp_use]
    
    logger.info('imp use {} {}'.format(imp_use.shape, n_features, drop_col))
    with open('features_tmp.py', 'w') as f:
        f.write('DROP_FEATURE = ["' + '", "'.join(map(str, drop_col)) + '"]\n')

    with open('features_use.py', 'w') as f:
        f.write('FEATURE = ["' + '", "'.join(map(str, drop_col)) + '"]\n')

