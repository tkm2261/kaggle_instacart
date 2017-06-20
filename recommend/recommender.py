import pandas
import numpy
from multiprocessing import Pool


def map_help(args):
    data, item_dic, ranking = args
    idx = numpy.argsort(data)[-ranking:]
    idx = [item_dic[i] for i in idx]

    return idx


class Recommender(object):

    def __init__(self,
                 recommender_data
                 ):

        self.recommender_data = recommender_data

        self.user_matrix = None
        self.item_matrix = None

        self.all_scores = None
        #self.user_item_link_matrix = None

    def fit(self):
        pass

    def predict(self, user_ids, ranking=100, index=False):

        if self.user_matrix is None or self.item_matrix is None:
            raise Exception("need to fit before predict.")

        if not index:
            user_idx = [self.recommender_data.map_user2idx[i] for i in user_ids]
        else:
            user_idx = user_ids

        score_matrix = numpy.dot(self.user_matrix[user_idx], self.item_matrix)

        score_matrix = pandas.DataFrame(score_matrix.T)
        """
        scores = []
        
        for user in score_matrix:
            #idx = score_matrix[user].rank(ascending=False)[:ranking]#.index.values
            #idx = score_matrix.sort(user, ascending=False).index.values
            idx = numpy.argsort(score_matrix[user])[-ranking:]
            idx = [self.recommender_data.map_idx2item[i] for i in idx]
            scores.append(list(idx))
        """
        p = Pool()
        scores = p.map(map_help, [[score_matrix[user],
                                   self.recommender_data.map_idx2item,
                                   ranking] for user in score_matrix])
        p.close()
        p.join()
        return scores

    def get_score(self, k=100, batch=10000):
        if self.all_scores is not None:
            return self.all_scores

        batch_num = int(self.user_matrix.shape[0] / batch) + 1

        scores = []
        for i in range(batch_num):
            print("batch:", i, batch_num)
            start = i * batch
            end = (i + 1) * batch

            if end > self.user_matrix.shape[0]:
                end = self.user_matrix.shape[0]

            scores += self.predict(range(start, end), ranking=k, index=True)

        self.all_scores = scores

        return scores
