import sys
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = pd.read_csv("ml-latest-small/ratings.csv")

user_ids = dataset['userId'].unique().tolist()
print(len(user_ids))
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids = dataset["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

dataset["user"] = dataset["userId"].map(user2user_encoded)
dataset["movie"] = dataset["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)

dataset["rating"] = dataset["rating"].values.astype(np.float32)

min_rating = min(dataset["rating"])
max_rating = max(dataset["rating"])

print(
    "Number of users: {}, Number of movies: {}, Min rating: {}, Max rating: {}".format(
        num_users,
        num_movies,
        min_rating,
        max_rating))


#########################################
# Prepare Training and Validation data
dataset = dataset.sample(frac=1)

x = dataset[["user", "movie"]].values
y = dataset["rating"].apply(lambda x: (
    x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.9 * dataset.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:])


x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train).float()
x_val=torch.from_numpy(x_val)
y_val=torch.from_numpy(y_val).float()

class DatasetColab(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        Y = self.y[index]

        return X, Y


#training_set = DatasetColab(x_train, y_train)
#val_set = DatasetColab(x_val, y_val)

#training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
#val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
EMBEDDING_SIZE = 50

###########################################
# Creating the PyTorch model


class RecommenderNet(LightningModule):
    def __init__(self, num_users, num_movies, embedding_size ):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.loss=nn.BCELoss()

        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)

        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = torch.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        x = nn.functional.sigmoid(x)
        return x

    def prepare_data(self):
        self.training_set=DatasetColab(x_train, y_train)
        self.val_set=DatasetColab(x_val, y_val)

    def train_dataloader(self):
        training_loader=DataLoader(self.training_set, batch_size=64, shuffle=True)
        return training_loader

    def val_dataloader(self):
        val_loader=DataLoader(self.val_set, batch_size=64, shuffle=False)
        return val_loader

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(), lr=0.001 ,  eps=1e-7)
        return optimizer

    def training_step(self, batch, batch_idx):
        features, targets=batch
        targets=targets.unsqueeze(1)
        outputs=self.forward(features)
        cost=self.loss(outputs, targets)
        log={'train_loss': cost}
        return {'loss':cost, 'log': log}

    def training_epoch_end(self, outputs):
        avg_loss=torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs={'avg_train_loss': avg_loss}
        return {'avg_train_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        features, targets=batch
        targets=targets.unsqueeze(1)
        outputs=self.forward(features)
        cost=self.loss(outputs, targets)
        return {'val_loss': cost}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log={'avg_loss': avg_loss}
        return {'avg_loss': avg_loss, 'log': log}



############################################################
model=RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
input_type=input("Type train to train the model or type predict to predict from pretrained model")
if input_type=="train":
    trainer=pl.Trainer(gpus=1, max_epoch=5)
    trainer.fit(model)
    trainer.save_checkpoint('example.ckpt')
else:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load('example.ckpt', map_location= lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    movie_df = pd.read_csv("ml-latest-small/movies.csv")
    df = dataset
    # Let us get a user and see the top recommendations.
    user_id=242
    #user_id = df.userId.sample(1).iloc[0]
    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[
        ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_id]] * len(movies_not_watched), movies_not_watched)
    )
    user_movie_array = torch.from_numpy(user_movie_array)
    ratings = model(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:]
    recommended_movie_ids = [movie_encoded2movie.get(
        movies_not_watched[x][0]) for x in top_ratings_indices]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("Movies with high ratings from user")
    print("----" * 8)
    top_movies_user = (
        movies_watched_by_user.sort_values(by="rating", ascending=False)
        .head(5)
        .movieId.values
    )
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
    for row in movie_df_rows.itertuples():
        print(row.title, ":", row.genres)

    print("----" * 8)
    print("Top 10 movie recommendations")
    print("----" * 8)
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    for row in recommended_movies.itertuples():
        print(row.title, ":", row.genres)
