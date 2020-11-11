from numpy.core.fromnumeric import size
import pandas as pd

import joblib
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer

import preprocessing as prep
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

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
    df['clean_comment'] = df['clean_comment'].map(lambda x: prep.preprocess(x))
    df_train, df_test = split(df, 0.33)

    df_train.to_csv('Project/data/train.csv')
    df_test.to_csv('Project/data/test.csv')
    X_train = df_train['clean_comment']
    y_train = df_train['category']

    vectorizer = TfidfVectorizer()
    model = LinearSVC()
    pipeline = make_pipeline(vectorizer, model)

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'Project/src/model/pipeline.pkl')

if __name__ == "__main__":
    train()
