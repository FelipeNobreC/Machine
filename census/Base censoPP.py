import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import pickle


class CensusDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_census = None
        self.X_census = None
        self.y_census = None
        self.label_encoders = {}
        self.column_transformer = None
        self.scaler = None

    def carregar_dados(self):
        # Carrega os dados do arquivo CSV e separa as variáveis dependentes e independentes.
        self.base_census = pd.read_csv(self.file_path)
        self.X_census = self.base_census.iloc[:, 0:14].values
        self.y_census = self.base_census.iloc[:, 14].values

    def aplicar_label_encoding(self):
        # Aplica Label Encoding nas colunas categóricas para converter strings em números.
        colunas_categoricas = [1, 3, 5, 6, 7, 8, 9, 13]
        for col in colunas_categoricas:
            le = LabelEncoder()
            self.X_census[:, col] = le.fit_transform(self.X_census[:, col])
            self.label_encoders[col] = le

    def aplicar_one_hot_encoding(self):
        # Aplica One-Hot Encoding nas colunas categóricas previamente codificadas.
        self.column_transformer = ColumnTransformer(
            transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
            remainder='passthrough'
        )
        self.X_census = self.column_transformer.fit_transform(self.X_census).toarray()

    def escalonar_dados(self):
        # Escalona os dados para que todas as variáveis tenham a mesma escala.
        self.scaler = StandardScaler()
        self.X_census = self.scaler.fit_transform(self.X_census)

    def dividir_dados(self, test_size=0.15, random_state=0):
        # Divide os dados em conjuntos de treinamento e teste.
        return train_test_split(self.X_census, self.y_census, test_size=test_size, random_state=random_state)

    def salvar_dados_processados(self, save_path, dados):
        # Salva os dados processados em um arquivo pickle para reutilização.
        with open(save_path, mode='wb') as f:
            pickle.dump(dados, f)

    def gerar_grafico_treemap(self):
        # Gera e exibe um gráfico treemap das colunas ocupação, relacionamento e idade.
        grafico = px.treemap(self.base_census, path=['occupation', 'relationship', 'age'])
        grafico.show()


if __name__ == "__main__":
    # Criação do objeto e processamento
    processor = CensusDataProcessor(file_path='./basededados/census.csv')

    # Carregar e processar dados
    processor.carregar_dados()
    processor.aplicar_label_encoding()
    processor.aplicar_one_hot_encoding()
    processor.escalonar_dados()

    # Divisão dos dados
    X_treino, X_teste, y_treino, y_teste = processor.dividir_dados()

    # Exibir formas dos dados
    print("Treinamento (X):", X_treino.shape)
    print("Teste (X):", X_teste.shape)
    print("Treinamento (y):", y_treino.shape)
    print("Teste (y):", y_teste.shape)

    # Salvar dados processados
    processor.salvar_dados_processados('census.pkl', [X_teste, X_treino, y_teste, y_treino])

    # Gerar gráfico treemap
    processor.gerar_grafico_treemap()
