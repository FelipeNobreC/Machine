import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle


class CensusClassifier:
    def __init__(self, model_path):
        # Inicializa o classificador com o caminho do modelo salvo e os dados.
        self.model_path = model_path
        self.X_census_treino = None
        self.X_census_teste = None
        self.y_census_treino = None
        self.y_census_teste = None
        self.svm = None

    def carregar_dados(self):
        # Carrega os dados processados salvos em um arquivo pickle.
        with open(self.model_path, 'rb') as f:
            self.X_census_treino, self.X_census_teste, self.y_census_treino, self.y_census_teste = pickle.load(f)
        
        # Garante que y_train está no formato correto
        self.y_census_treino = self.y_census_treino.ravel()

    def treinar_modelo(self, kernel='linear', random_state=1):
        # Treina o modelo SVM com os dados de treinamento.
        self.svm = SVC(kernel=kernel, random_state=random_state)
        self.svm.fit(self.X_census_treino, self.y_census_treino)

    def avaliar_modelo(self):
        # Avalia o modelo com os dados de teste.
        previsoes = self.svm.predict(self.X_census_teste)
        print(previsoes)
        print(accuracy_score(self.y_census_teste, previsoes))
        print(classification_report(self.y_census_teste, previsoes))

    def consultar(self, *valores):
        # Faz previsões para novos dados fornecidos.
        entrada = self.svm.fit(valores)
        previsao = self.svm.predict(entrada)
        print(accuracy_score(self.y_test, previsao))


if __name__ == "__main__":
    # Instancia o classificador
    classifier = CensusClassifier(model_path='/Users/felip/PycharmProjects/Machine/census/census.pkl')

    # Carregar dados
    classifier.carregar_dados()

    # Treinar modelo
    classifier.treinar_modelo()

    # Avaliar modelo
    classifier.avaliar_modelo()

