import scipy.sparse as spMat
import pandas
import numpy
from data import RecommenderData

COLUMN_NAMES = ["user_id", "product_id", 'score']


def read_csv(path,  delimiter=","):

    data = pandas.read_csv(path,
                           usecols=COLUMN_NAMES)

    user_ids = numpy.sort(data["user_id"].unique())
    item_ids = numpy.sort(data["product_id"].unique())

    map_idx2user = dict([(i, user_ids[i]) for i in range(len(user_ids))])
    map_idx2item = dict([(i, item_ids[i]) for i in range(len(item_ids))])
    map_user2idx = dict([(user_ids[i], i) for i in range(len(user_ids))])
    map_item2idx = dict([(item_ids[i], i) for i in range(len(item_ids))])

    data["user_id"] = data["user_id"].apply(lambda x: map_user2idx[x])
    data["product_id"] = data["product_id"].apply(lambda x: map_item2idx[x])

    data = spMat.coo_matrix(
        (data["score"], (data["user_id"], data["product_id"])),
        shape=(len(user_ids), len(item_ids)),
        dtype=numpy.double
    )

    return RecommenderData(data,
                           map_idx2user,
                           map_idx2item,
                           map_user2idx,
                           map_item2idx)
