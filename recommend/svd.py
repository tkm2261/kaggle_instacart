from recommender import Recommender
from sklearn.neighbors import NearestNeighbors, LSHForest
import numpy
import pandas
import scipy.sparse as spMat
from tqdm import tqdm


class SVD(Recommender):

    def fit(self, k=50, max_iter=15):
        if self.recommender_data.preference_matrix.shape[1] < k:
            k = self.recommender_data.preference_matrix.shape[1] - 1

        u, s, vt = spMat.linalg.svds(self.recommender_data.preference_matrix, k)
        #s = numpy.sqrt(s)
        u = u * s
        #vt = (vt.T * s).T
        self.user_matrix = u  # numpy.array([row / row.sum() for row in u])
        self.item_matrix = vt.T  # numpy.array([row / row.sum() for row in vt.T])

    def get_score(self, k=00, batch=10000):

        #nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='').fit(self.item_matrix)
        nbrs = LSHForest(n_neighbors=k, random_state=0).fit(self.item_matrix)
        print('fit')

        batch_num = int(self.user_matrix.shape[0] / batch) + 1

        map_scores = {}
        for i in tqdm(range(batch_num)):
            start = i * batch
            end = (i + 1) * batch

            if end > self.user_matrix.shape[0]:
                end = self.user_matrix.shape[0]

            dist, ind = nbrs.kneighbors(self.user_matrix[start:end], return_distance=True)
            for t, user_idx in enumerate(range(start, end)):
                map_scores[self.recommender_data.map_idx2user[user_idx]] = \
                    [{'item_id': self.recommender_data.map_idx2item[ind[t, j]], 'dist': dist[t, j]}
                        for j in range(dist.shape[1])]

        return map_scores
