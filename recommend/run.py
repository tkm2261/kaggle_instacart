from load_data import read_csv
from svd import SVD
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
if __name__ == '__main__':
    rec_data = read_csv('../input/user_item_cnt.csv')
    rec = SVD(rec_data)
    rec.fit()

    score = rec.get_score()
    tmp = [dict(v, user_id=user_id) for user_id, aaa in score.items() for v in aaa]
    df = pd.DataFrame(tmp)
    df.head()
    df.to_csv('svd2.csv', index=False)
    """
    model = MiniBatchKMeans(n_clusters=100, random_state=0)
    model.fit(rec.user_matrix)
    pred = model.predict(rec.user_matrix)

    users = [rec_data.map_idx2user[i] for i in range(len(rec_data.map_idx2user))]
    max(users)
    len(rec_data.map_idx2user)

    df = pd.DataFrame({'user_id': users, 'cluster': pred})
    df.to_csv('cluster.csv', index=False)
    """
