import pandas as pd

def get_data():
    d_train = pd.read_csv("data/train.csv")
    d_test = pd.read_csv("data/test.csv")

    return d_train, d_test


