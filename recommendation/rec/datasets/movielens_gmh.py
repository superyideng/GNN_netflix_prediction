import pandas as pd
import dgl
import os
import torch
import numpy as np
import scipy.sparse as sp
import time
from functools import partial
from .. import randomwalk
import stanfordnlp
import re
import tqdm
import string

class MovieLens(object):
    def __init__(self, directory):
        '''
        directory: path to movielens directory which should have the three
                   files:
                   users.dat
                   movies.dat
                   ratings.dat
        '''
        self.directory = directory

        users = []
        movies = []
        ratings = []

        # read users
        with open(os.path.join(directory, 'users_10000.dat')) as f:
            for l in f:
                id_, gender = l.strip().split('::')
                users.append({
                    'id': int(id_),
                    'gender': gender,
                    })
        self.users = pd.DataFrame(users).set_index('id').astype('category')

        # read movies
        with open(os.path.join(directory, 'movies_2000.dat'), encoding='latin1') as f:
            for l in f:
                id_, title = l.strip().split('::')

                # extract year
                try:
                    assert re.match(r'.*\([0-9]{4}\)$', title)
                    year = title[-5:-1]
                    title = title[:-6].strip()
                except Exception as e:
                    title = title[:-6].strip()
                    year = '2020'

                data = {'id': int(id_), 'title': title, 'year': year}
                movies.append(data)
        self.movies = (
                pd.DataFrame(movies)
                .set_index('id')
                .fillna(False)
                .astype({'year': 'category'}))
        self.genres = self.movies.columns[self.movies.dtypes == bool]

        # read ratings
        with open(os.path.join(directory, 'ratings_all.dat')) as f:
            for l in f:
                user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        ratings = pd.DataFrame(ratings)
        ratings = ratings[ratings['user_id'] isin (self.users.index)]
        ratings = ratings[ratings['user_id'] isin (self.movie.index)]
        user_cnt = pd.DataFrame(ratings.groupby('user_id')['rating'].count()).reset_index()
        user_cnt.columns=['user_id','user_cnt']
        ratings = pd.merge(ratings, user_cnt, on=['user_id', 'user_id'])
        ratings = ratings[ratings.user_cnt>=2]
        ratings = ratings.reset_index()
        movie_count = ratings['movie_id'].value_counts()
        movie_count.name = 'movie_count'
        ratings = ratings.join(movie_count, on='movie_id')
        self.ratings = ratings

        #print(self.movies.head())
        #print(self.users.head())
        #print(self.ratings.head())
        #print(len(self.ratings))
        set_user = set(self.users.index)
        set_ratings_user = set(self.ratings['user_id'])
        #print("===========================================================")
        #print(set_ratings_user-set_user)
        #print(self.ratings['user_id'][0])
        #print(self.users.index[0])
        # drop users and movies which do not exist in ratings
        self.users = self.users[self.users.index.isin(self.ratings['user_id'])]
        self.movies = self.movies[self.movies.index.isin(self.ratings['movie_id'])]
        #self.ratings = self.ratings[self.ratings['user_id'].isin(self.users.index)]
        #self.ratings = self.ratings[self.ratings['movie_id'].isin(self.movies.index)]
        #print("----")
        #print(len(self.ratings))

        #print(self.movies.head())
        #print(self.users.head())
        #print(self.ratings.head())
        self.data_split()
        self.build_graph()

    def split_user(self, df, filter_counts=False):
        df_new = df.copy()
        df_new['prob'] = 0

        if filter_counts:
            df_new_sub = (df_new['movie_count'] >= 10).nonzero()[0]
        else:
            df_new_sub = df_new['train'].nonzero()[0]
        prob = np.linspace(0, 1, df_new_sub.shape[0],endpoint=True)
        np.random.shuffle(prob)
        df_new['prob'].iloc[df_new_sub] = prob
        return df_new

    def data_split(self):
        self.ratings = self.ratings.groupby('user_id', group_keys=False).apply(
                partial(self.split_user, filter_counts=True))
        self.ratings['train'] = self.ratings['prob'] <= 0.9
        #self.ratings['valid'] = (self.ratings['prob'] > 0.8) & (self.ratings['prob'] <= 0.9)
        self.ratings['test'] = self.ratings['prob'] > 0.9
        self.ratings.drop(['prob'], axis=1, inplace=True)
        print(self.ratings)

    def build_graph(self):
        user_ids = list(self.users.index)
        movie_ids = list(self.movies.index)
        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        movie_ids_invmap = {id_: i for i, id_ in enumerate(movie_ids)}
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.user_ids_invmap = user_ids_invmap
        self.movie_ids_invmap = movie_ids_invmap

        g = dgl.DGLGraph()
        g.add_nodes(len(user_ids) + len(movie_ids))

        # user features
        for user_column in self.users.columns:
            udata = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
            # 0 for padding
            udata[:len(user_ids)] = \
                    torch.LongTensor(self.users[user_column].cat.codes.values.astype('int64') + 1)
            g.ndata[user_column] = udata


        # movie year
        g.ndata['year'] = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
        # 0 for padding
        g.ndata['year'][len(user_ids):len(user_ids) + len(movie_ids)] = \
                torch.LongTensor(self.movies['year'].cat.codes.values.astype('int64') + 1)

        # movie title
        nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize,lemma')
        vocab = set()
        title_words = []
        for t in tqdm.tqdm(self.movies['title'].values):
            doc = nlp(t)
            words = set()
            for s in doc.sentences:
                words.update(w.lemma.lower() for w in s.words
                             if not re.fullmatch(r'['+string.punctuation+']+', w.lemma))
            vocab.update(words)
            title_words.append(words)
        vocab = list(vocab)
        vocab_invmap = {w: i for i, w in enumerate(vocab)}
        # bag-of-words
        g.ndata['title'] = torch.zeros(g.number_of_nodes(), len(vocab))
        for i, tw in enumerate(tqdm.tqdm(title_words)):
            g.ndata['title'][len(user_ids) + i, [vocab_invmap[w] for w in tw]] = 1
        self.vocab = vocab
        self.vocab_invmap = vocab_invmap
        print(self.ratings['user_id'].values)
        rating_user_vertices = [user_ids_invmap[id_] for id_ in self.ratings['user_id'].values]
        rating_movie_vertices = [movie_ids_invmap[id_] + len(user_ids)
                                 for id_ in self.ratings['movie_id'].values]
        self.rating_user_vertices = rating_user_vertices
        self.rating_movie_vertices = rating_movie_vertices

        g.add_edges(
                rating_user_vertices,
                rating_movie_vertices,
                data={'inv': torch.zeros(self.ratings.shape[0], dtype=torch.uint8)})
        g.add_edges(
                rating_movie_vertices,
                rating_user_vertices,
                data={'inv': torch.ones(self.ratings.shape[0], dtype=torch.uint8)})
        self.g = g

    def generate_mask(self):
        while True:
            ratings = self.ratings.groupby('user_id', group_keys=False).apply(self.split_user)
            #print(ratings)
            prior_prob = ratings['prob'].values
            for i in range(5):
                train_mask = (prior_prob >= 0.2 * i) & (prior_prob < 0.2 * (i + 1))
                prior_mask = (prior_prob >=0)
                train_mask &= ratings['train'].values
                prior_mask &= ratings['train'].values

                if np.sum(prior_mask)==0 or np.sum(train_mask)==0:
                    continue
                yield prior_mask, train_mask

    def refresh_mask(self):
        if not hasattr(self, 'masks'):
            self.masks = self.generate_mask()
        prior_mask, train_mask = next(self.masks)

        #valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(train_mask.astype('uint8'))
        prior_tensor = torch.from_numpy(prior_mask.astype('uint8'))
        edge_data = {
                'prior': prior_tensor,
                'test': test_tensor,
                'train': train_tensor,
                }

        self.g.edges[self.rating_user_vertices, self.rating_movie_vertices].data.update(edge_data)
        self.g.edges[self.rating_movie_vertices, self.rating_user_vertices].data.update(edge_data)
