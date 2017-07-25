import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import re
from IPython.core.display import display
from tqdm import tqdm


df = pd.read_csv('../input/df_prior.csv', usecols=['product_id', 'product_name', 'reordered'])
print('load')


def aaa(x):
    ret = re.sub(r'(\(.*?\)|[^a-z^A-Z])', ' ', x.lower()).split()
    return ret


tmp = df['product_name'].apply(aaa)

#from gensim import models
from gensim import corpora
from gensim.matutils import corpus2csc

dictionary = corpora.Dictionary(tmp)
dictionary.filter_extremes(no_below=2, no_above=1., keep_n=2000000)
print('dict')
id_corpus = map(dictionary.doc2bow, tmp)

count_mat = corpus2csc(id_corpus).T
print('csc')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

"""
model = LogisticRegression(random_state=0)
model.fit(count_mat, df.reordered.values)
pred = model.predict_proba(count_mat)[:, 1]
df['pred_logreg'] = pred
print('aaa')
"""
model = LGBMClassifier(max_depth=3, subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
                       seed=0, n_estimators=1000, learning_rate=0.1, reg_alpha=0.1)
model.fit(count_mat, df.reordered.values)
pred = model.predict_proba(count_mat)[:, 1]
df['pred_gbm'] = pred


model = BernoulliNB()
model.fit(count_mat, df.reordered.values)
pred = model.predict_proba(count_mat)[:, 1]
df['pred_naive'] = pred
print('bbb')

df[['product_id', 'pred_naive', 'pred_gbm']].drop_duplicates().to_csv('word_preds.csv', index=False)
