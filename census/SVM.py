import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt


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
        return previsoes

    def mostrar_grafico_pizza(self, previsoes):
        # Gera um gráfico de pizza para visualizar as proporções de cada classe nas previsões.
        classes, contagens = np.unique(previsoes, return_counts=True)
        plt.figure(figsize=(8, 8))
        plt.pie(
            contagens,
            labels=classes,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Paired.colors
        )
        plt.title("Distribuição das Previsões")
        plt.show()


if __name__ == "__main__":
    # Instancia o classificador
    classifier = CensusClassifier(model_path='/Users/felip/PycharmProjects/scripts/Machine/census/census.pkl')

    # Carregar dados
    classifier.carregar_dados()

    # Treinar modelo
    classifier.treinar_modelo()

    # Avaliar modelo
    previsoes = classifier.avaliar_modelo()

    # Mostrar gráfico de pizza com as previsões
    classifier.mostrar_grafico_pizza(previsoes)
