import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

with open('/Users/felip/PycharmProjects/MA e Data/census/census.pkl', 'rb') as f:
  X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = pickle.load(f)


# Verifica a forma original do y_census_treinamento
#print(y_census_treinamento.shape)
print(X_census_treinamento)
# Reshape para uma dimensão
y_census_treinamento = y_census_treinamento.ravel()

svm_census = SVC(kernel='linear', random_state=1)
svm_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = svm_census.predict(X_census_teste)
print(previsoes)
print(accuracy_score(y_census_teste, previsoes))



print(classification_report(y_census_teste, previsoes))


def consulta(*valores):
  # Certifica-se de que os valores estão em um formato adequado para previsão
  entrada = svm_census.fit(valores)
  prev = svm_census.predict(entrada)
  print(accuracy_score(y_census_teste, prev))