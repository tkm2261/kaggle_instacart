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

"""
def load_wordvec():

    f = open('vectors.txt', 'r')
    map_result = {}
    f.readline()
    for line in f:
        line = line.strip().split(' ')
        word = line[0].lower()
        vec = np.array(list(map(float, line[1:])))
        map_result[word] = vec
    return map_result

map_vec = load_wordvec()

with open('map_vec.pkl', 'wb') as f:
    pickle.dump(map_vec, f, -1)
"""

na = np.ones(100) * -1

with open('map_vec.pkl', 'rb') as f:
    map_vec = pickle.load(f)

with open('train_data.pkl', 'rb') as f:
    df = pickle.load(f)

df = df['o_product_id'].values
df = np.vstack([map_vec.get(key, na) for key in df])

with open('train_word.pkl', 'wb') as f:
    pickle.dump(df, f, -1)

with open('test_data.pkl', 'rb') as f:
    df = pickle.load(f)

df = df['o_product_id'].values
df = np.vstack([map_vec.get(key, na) for key in df])
with open('test_word.pkl', 'wb') as f:
    pickle.dump(df, f, -1)
