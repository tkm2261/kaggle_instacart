import pandas as pd
import numpy as np
import pickle
from flurs.recommender.user_knn import UserKNNRecommender
from flurs.data.entity import User, Item, Event
from tqdm import tqdm
recommender = UserKNNRecommender()

recommender.init_params()
recommender.init_recommender()


df = pd.read_csv('../input/df_prior.csv', usecols=['user_id', 'product_id']).values

map_user = {}
map_item = {}
user_idx = 0
item_idx = 0

print('load')
for user, item in tqdm(df):
    if user not in map_user:
        map_user[user] = user_idx
        user_idx += 1
    if item not in map_item:
        map_item[item] = item_idx
        item_idx += 1
    user = map_user[user]
    item = map_item[item]

    user = User(user)
    recommender.add_user(user)

    item = Item(item)
    recommender.add_item(item)

    event = Event(user, item, 1)
    recommender.update(event)

with open('recommend.pkl', 'wb') as f:
    pickle.dump(recommender, f, -1)
# specify target user and list of item candidates
#recommender.recommend(user, [0])
# => (sorted candidates, scores)
