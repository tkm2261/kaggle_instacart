import scipy.sparse as spMat
import pandas
import numpy
from data import RecommenderData
import pickle
from tqdm import tqdm
import os
from multiprocessing import Pool    
COLUMN_NAMES = ["order_id", "product_id", "user_id", "reordered"]

path = '../input/df_prior.csv'
df = pandas.read_csv(path,
                         usecols=COLUMN_NAMES, dtype=int)

def read_csv(data,  delimiter=","):
    #a = data['reordered'].values
    data['score'] = numpy.ones(data.shape[0], dtype=numpy.float64)
    order_ids = numpy.sort(data["order_id"].unique())
    item_ids = numpy.sort(data["product_id"].unique())

    map_idx2user = dict([(i, order_ids[i]) for i in range(len(order_ids))])
    map_idx2item = dict([(i, item_ids[i]) for i in range(len(item_ids))])
    map_user2idx = dict([(order_ids[i], i) for i in range(len(order_ids))])
    map_item2idx = dict([(item_ids[i], i) for i in range(len(item_ids))])

    data["order_id"] = data["order_id"].apply(lambda x: map_user2idx[x])
    data["product_id"] = data["product_id"].apply(lambda x: map_item2idx[x])
    A = spMat.coo_matrix(
        (data["score"], (data["order_id"], data["product_id"])),
        shape=(len(order_ids), len(item_ids))
    ).T.tolil()
    print(0)    
    mu = A.mean(axis=1)
    print(1, mu.shape)
    N = A.shape[1]
    print(2)
    for i, m in tqdm(enumerate(mu)):
        A.data[i] = [t - m[0, 0] for t in A.data[i]]
    print(3, A.shape)    
    A = A * A.T
    print(4)
    A /= N
    print(5)    
    """
    C = ((A.T * A - (sum(A).T * sum(A) / N)) / (N - 1)).todense()

    V = numpy.sqrt(numpy.mat(numpy.diag(C)).T * numpy.mat(numpy.diag(C)))
    COV = numpy.divide(C, V + 1e-119)
    """
    #COV = numpy.cov(A)
    return RecommenderData(data,
                           A,
                           map_idx2user,
                           map_idx2item,
                           map_user2idx,
                           map_item2idx)

def aaa(user_id):
    if os.path.exists('cov_data4/%s.pkl' % (user_id)):
        return
    aaa = read_csv(df[df["user_id"] == user_id].copy())
    with open('cov_data4/%s.pkl' % (user_id), 'wb') as f:
        pickle.dump(aaa, f, -1)
    

if __name__ == '__main__':

    a = read_csv(df)
    with open('cov_data/all_cov.pkl', 'wb') as f:
        pickle.dump(a, f, -1)
