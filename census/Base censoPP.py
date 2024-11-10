import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


base_census = pd.read_csv('basededados/census.csv')
X_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values


#labelencoder transforma os valores string em num
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()
X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])


#onehotencoder transforma valores das colunas em valores 'encriptados'
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
#salva a tabela x formatada e adpta para o formato do numpy
X_census = onehotencoder_census.fit_transform(X_census).toarray()

#escalonamento deixa os valores na mesma escala
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)
print(X_census)
#print(X_census.shape)
#print(base_census)

#divisao da base de dados em base de treinamento e teste

X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census, y_census, test_size = 0.15, random_state = 0)

#salvando as variaveis para n ter que rodar
# todo esse codigo dnv

grafico = px.treemap(base_census, path=['occupation', 'relationship', 'age'])
grafico.show()
print(y_census_treinamento.shape)
print(y_census_teste.shape)
print(X_census_treinamento.shape)
print(X_census_teste.shape)

with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_teste, X_census_treinamento, y_census_teste, y_census_treinamento], f)

