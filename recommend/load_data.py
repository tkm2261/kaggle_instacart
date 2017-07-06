import scipy.sparse as spMat
import pandas
import numpy
from data import RecommenderData

COLUMN_NAMES = ["order_id", "product_id", "user_id"]


def read_csv(data,  delimiter=","):

    data['score'] = numpy.ones(data.shape[0], dtype=numpy.int8)
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
    ).todense().T
    """
    N = A.shape[1]
    C = ((A.T * A - (sum(A).T * sum(A) / N)) / (N - 1)).todense()

    V = numpy.sqrt(numpy.mat(numpy.diag(C)).T * numpy.mat(numpy.diag(C)))
    COV = numpy.divide(C, V + 1e-119)
    """
    COV = numpy.cov(A)
    return RecommenderData(data,
                           COV,
                           map_idx2user,
                           map_idx2item,
                           map_user2idx,
                           map_item2idx)


if __name__ == '__main__':
    path = '../input/df_prior.csv'
    data = pandas.read_csv(path,
                           usecols=COLUMN_NAMES, dtype=int)

    #aaa = read_csv(data)
    import pickle
    from tqdm import tqdm
    # with open('cov_data/all_cov.pkl', 'wb') as f:
    #    pickle.dump(aaa, f, -1)
    import os
    ids = data.user_id.unique()
    for user_id in tqdm(ids):
        if os.path.exists('cov_data/%s.pkl' % (user_id)):
            continue
        aaa = read_csv(data[data["user_id"] == user_id].copy())
        with open('cov_data/%s.pkl' % (user_id), 'wb') as f:
            pickle.dump(aaa, f, -1)
