from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class CreditDataProcessor1:
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_credit = None
        self.x_credit = None
        self.y_credit = None
        self.scale_credit = StandardScaler()
        self.cria_rn()

    def load_new_data(self):
        # Carregar arquivo CSV
        self.base_credit = pd.read_csv(self.file_path)

    def cria_rn(self):
        self.rede_neural = MLPClassifier(
            max_iter=5000, verbose=True, tol=1e-5,  # Aumentar max_iter para garantir mais iterações
            solver='adam', activation='relu', hidden_layer_sizes=(20, 20)
        )
        self._load_data()

    def _load_data(self):
        # Carrega os dados de treinamento e teste do arquivo pickle.
        with open('/Users/felip/PycharmProjects/scripts/Machine/credit/credit.pkl', 'rb') as f:
            self.X_train, self.X_test, self.y_train, self.y_test = pickle.load(f)
        self.scale_credit.fit(self.X_train)  # Ajusta o escalonador aos dados de treinamento

    def preprocess_data(self):
        self.x_credit = self.base_credit.iloc[:, 1:4].values
        self.y_credit = self.base_credit.iloc[:, 4].values
        print(self.x_credit.shape)
        print(self.base_credit.columns)
        # Dimensionar as variáveis ​​preditivas
        self.x_credit = self.scale_credit.transform(self.x_credit)

    def train_model(self):
        # Treina a rede neural com os dados de treinamento.
        print("Treinando o modelo...")
        self.rede_neural.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Avalia o modelo com os dados de teste.
        #scores = cross_val_score(self.rede_neural, self.X_train, self.y_train, cv=5)
        #print("Média da acurácia com validação cruzada:", scores.mean())
        predictions = self.rede_neural.predict(self.x_credit)
        print(predictions)



if __name__ == "__main__":
    credit_nn = CreditDataProcessor1('/Users/felip/PycharmProjects/scripts/Machine/basededados/Pasta 10(Planilha1) 2.csv')
    credit_nn.load_new_data()
    credit_nn.preprocess_data()

    # Treinar e avaliar o modelo
    credit_nn.train_model()
    credit_nn.evaluate_model()



