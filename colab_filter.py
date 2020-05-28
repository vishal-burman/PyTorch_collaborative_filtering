import pandas as pd
import torch
import numpy as np
import torch.nn as nn
dataset=pd.read_csv("ml-latest-small/ratings.csv")

user_ids=dataset['userId'].unique().tolist()
print(len(user_ids))
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids=dataset["movieId"].unique().tolist()
movie2movie_encoded={x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie={i: x for i, x in enumerate(movie_ids)}

dataset["user"]=dataset["userId"].map(user2user_encoded)
dataset["movie"]=dataset["movieId"].map(movie2movie_encoded)

num_users=len(user2user_encoded)
num_movies=len(movie_encoded2movie)

dataset["rating"]=dataset["rating"].values.astype(np.float32)

min_rating=min(dataset["rating"])
max_rating=max(dataset["rating"])

print("Number of users: {}, Number of movies: {}, Min rating: {}, Max rating: {}".format(num_users, num_movies, min_rating, max_rating))


#########################################
# Prepare Training and Validation data
dataset=dataset.sample(frac=1, random_state=42)
x=dataset[["user", "movie"]].values
y=dataset["rating"].apply(lambda x: (x-min_rating)/(max_rating-min_rating)).values

train_indices=int(0.9* dataset.shape[0])
x_train, x_val, y_train, y_val=(x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:])
print(x_train[:5])
print(y_train[:5])
print(x_train[:, 0])
###########################################
"""
# Creating the PyTorch model
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users=num_users
        self.num_movies=num_movies
        self.embedding_size=embedding_size
        
        self.user_embedding=nn.Embedding(num_users, embedding_size)
        self.user_bias=nn.Embedding(num_users, 1)
        
        self.movie_embedding=nn.Embedding(num_movies, embedding_size)
        self.movie_bias=nn.Embedding(num_movies, 1)

    def forward(self, inputs):
        user_vector=self.user_embedding(inputs[:, 0])
        user_bias=self.user_bias(inputs[:, 0])
        movie_vector=self.movie_embedding(inputs[:, 1])
        movie_bias=self.movie_bias(inputs[:, 1])
        dot_user_movie=torch.tensordot(user_vector, movie_vector, 2)
        x=dot_user_movie+user_bias+movie_bias
        x=nn.functional.sigmoid(x)
        return x

"""
