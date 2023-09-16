import pandas as pd
from params import *

def process(row, column, weight):
    if row[column] == 0:
        return pd.concat([row] + [row.copy() for _ in range(weight - 1)], axis=1)
    return row

def resample(train_data):
    for organ, weight in OVERSAMPLING_WEIGHTS.items():
        train_data = pd.concat([process(row, f'{organ}_healthy', weight) for _, row in train_data.iterrows()], axis=1).T
    return train_data