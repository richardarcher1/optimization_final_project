local_ = False

if local_:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import json
# import pandas as pd
# import polars as pl
import unicodedata

import polars as pl
import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset#, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    df_train = pl.read_csv("data/1_train_test_split/df_train.csv")
    df_test = pl.read_csv("data/1_train_test_split/df_test.csv")
    df_validation = pl.read_csv("data/1_train_test_split/df_validation.csv")

    train_texts = df_train["text"].to_list()
    train_labels = df_train["stars"].to_list()

    test_texts = df_test["text"].to_list()
    test_labels = df_test["stars"].to_list()

    val_texts = df_validation["text"].to_list()
    val_labels = df_validation["stars"].to_list()

    # embed text
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = "intfloat/multilingual-e5-large-instruct"
    print("loading embedder")
    embedder = SentenceTransformer(model_name, device="cuda")

    print("embedding train dataset")
    train_embeddings = embedder.encode(train_texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True)

    print("embedding test dataset")
    test_embeddings = embedder.encode(test_texts, batch_size=128, convert_to_numpy=True,show_progress_bar=True)

    print("embedding val dataset")
    val_embeddings = embedder.encode(val_texts, batch_size=128, convert_to_numpy=True,show_progress_bar=True)

    class StarRatingDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = torch.tensor(self.X[idx], dtype=torch.float32)
            # Convert label to a one-hot vector: label in {1,...,5} -> one-hot of length 5
            y_onehot = torch.zeros(5)
            y_onehot[self.y[idx] - 1] = 1.0
            return x, y_onehot

    print("asldfkj")
    train_dataset = StarRatingDataset(train_embeddings, train_labels)
    test_dataset = StarRatingDataset(test_embeddings, test_labels)
    val_dataset = StarRatingDataset(val_embeddings, val_labels)

    # torch.save(train_dataset, '../data/2_training_ready/embedding00/train_dataset00.pth')
    # torch.save(test_dataset, '../data/2_training_ready/embedding00/test_dataset00.pth')
    # torch.save(val_dataset, '../data/2_training_ready/embedding00/val_dataset00.pth')

    torch.save(train_dataset, 'data/2_training_ready/embedding00/train_dataset01.pth')
    torch.save(test_dataset, 'data/2_training_ready/embedding00/test_dataset01.pth')
    torch.save(val_dataset, 'data/2_training_ready/embedding00/val_dataset01.pth')

    return 1

if __name__ == '__main__':
    main()