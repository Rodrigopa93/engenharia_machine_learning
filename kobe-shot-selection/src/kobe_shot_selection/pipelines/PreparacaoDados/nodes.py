"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from kedro_mlflow.io.metrics import MlflowMetricDataSet


def select_data(data: pd.DataFrame):
    data = data[['lat','lon','minutes_remaining','period','playoffs','shot_distance','shot_type','shot_made_flag']]
    return data

def filter_data(data: pd.DataFrame, shot_type_filter):
    data_filtered = data[data['shot_type'] == shot_type_filter]
    data_filtered = data_filtered.dropna()
    return data_filtered

def split_data(data: pd.DataFrame, test_size, random_state):

    X = data.drop('shot_made_flag', axis=1)  
    y = data['shot_made_flag'] 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def split_metrics(X_train, X_test, test_size):
    metrics = {
        'train_df_size': X_train.shape[0],
        'test_df_size': X_test.shape[0],
        'test_size_perc': test_size
    }

    return {
        key: {'value': float(value) if isinstance(value, (int, float)) else value, 'step': 1}
        for key, value in metrics.items()
    }