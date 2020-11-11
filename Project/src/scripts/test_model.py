from re import X
import joblib
import model_building as mb
import pandas as pd

import joblib


def test_model_accuracy():
    mb.train()
    model = joblib.load("Project/src/model/pipeline.pkl")
    test_df = pd.read_csv("Project/data/test.csv").dropna()
    X = test_df['clean_comment']
    y = test_df['category']
    score = model.score(X, y)
    assert score > 0.80