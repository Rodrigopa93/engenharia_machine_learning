# Importar bibliotecas
import pandas as pd
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

data_kobe = pd.read_csv('../data/01_raw/data.csv')

data = data_kobe[data_kobe['shot_type'] == '2PT Field Goal']
data = data[['lat','lon','minutes_remaining','period','playoffs','shot_distance','shot_made_flag']]
data = data.dropna()
data

target_names = data.columns

X = data.drop('shot_made_flag', axis=1)
y = data['shot_made_flag'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = AdaBoostClassifier(n_estimators=100, random_state=0)
model.fit(X, y)