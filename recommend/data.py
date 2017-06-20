import scipy.sparse as spMat

class RecommenderData(object):

    def __init__(self,
                 preference_matrix=None,
                 map_idx2user=None,
                 map_idx2item=None,
                 map_user2idx=None,
                 map_item2idx=None
                ):

        if spMat.issparse(preference_matrix):
            self.preference_matrix = preference_matrix
        else:
            self.preference_matrix = spMat.csr_matrix(preference_matrix)

        self.map_idx2user = map_idx2user
        self.map_idx2item = map_idx2item
        self.map_user2idx = map_user2idx
        self.map_item2idx = map_item2idx
