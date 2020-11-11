from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
from pandas.core.frame import DataFrame
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def split(df:DataFrame, test_size):
    sizes = df.shape
    
    df = df.sample(frac=1)
    if 0 < test_size < 1:
        size = (1 - test_size) * sizes[0]
        size = int(size)
        return df.iloc[:size, :], df.iloc[size:, :]
    else:
        return df


def train():
    df = pd.read_csv('Project/data/Reddit_Data.csv')

    df = df.dropna().reset_index(drop=True)

    df_train, df_test = split(df, 0.33)

    df_train.to_csv('Project/data/train.csv')
    df_test.to_csv('Project/data/test.csv')
    X_train = df_train['clean_comment']
    X_test = df_test['clean_comment']
    y_train = df_train['category']
    y_test = df_test['category']

    vectorizer = TfidfVectorizer()
    model = LinearSVC()
    pipeline = make_pipeline(vectorizer, model)

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'Project/src/model/pipeline.pkl')

if __name__ == "__main__":
    train()
