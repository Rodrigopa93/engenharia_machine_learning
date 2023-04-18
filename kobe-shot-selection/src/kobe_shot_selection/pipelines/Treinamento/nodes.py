"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

import pandas as pd
from pycaret.classification import *
from sklearn.metrics import log_loss, f1_score
from kedro_mlflow.io.metrics import MlflowMetricDataSet

# Treina um modelo de regressão logística com pyCaret
def train_lr_model(x_train, y_train, session_id):

    # inicializa a configuração do pyCaret com os dados de treinamento e identificador de sessão
    setup(data=x_train, target=y_train, session_id=session_id)

    # treina o modelo de regressão logística
    lr_model = create_model('lr')

    return lr_model


# Calcula a métrica log loss para um modelo de regressão logística
def compute_lr_metrics(model, x_test: pd.DataFrame, y_test):
    # faz a previsão com o modelo no conjunto de teste
    test_predictions = predict_model(model, data=x_test)
    # extrai o valor real do conjunto de teste
    test_y = y_test
    # calcula a métrica log loss entre a previsão e os valores reais
    log_loss_value = log_loss(test_y, test_predictions['prediction_label'])

    metrics = {'log_loss_lr': log_loss_value}

    return {
        key: {'value': value, 'step': 1}
        for key, value in metrics.items()
    }


# Treina um modelo AdaBoost com pyCaret
def train_ada_model(x_train: pd.DataFrame, y_train, session_id):
    # inicializa a configuração do pyCaret com os dados de treinamento e identificador de sessão
    setup(data=x_train, target=y_train, session_id=session_id)

    # treina o modelo AdaBoost com os hiperparâmetros padrão
    nb_model = create_model('ada')

    return nb_model

# Calcula as métricas log loss e F1 score para um modelo AdaBoost
def compute_ada_metrics(model, x_test: pd.DataFrame, y_test):
    # faz a previsão com o modelo no conjunto de teste
    test_predictions = predict_model(model, data=x_test)
    # extrai o valor real do conjunto de teste
    test_y = y_test
    # calcula as métricas log loss e F1 score entre a previsão e os valores reais
    log_loss_value = log_loss(test_y, test_predictions['prediction_label'])
    f1_score_value = f1_score(test_y, test_predictions['prediction_label'])
    
    metrics = {
        'log_loss_ada': log_loss_value, 
        'f1_score_ada': f1_score_value
               }

    # retorna as métricas com um dicionário em um formato adequado para o MLflow
    return {
        key: {'value': value, 'step': 1}
        for key, value in metrics.items()
    }
