import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

class CreditDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_credit = None
        self.x_credit = None
        self.y_credit = None
        self.scale_credit = StandardScaler()
        self.X_credit_treinamento = None
        self.X_credit_teste = None
        self.Y_credit_treinamento = None
        self.Y_credit_teste = None
    
    def load_data(self):
        # Carregar arquivo CSV
        self.base_credit = pd.read_csv(self.file_path)

    def preprocess_data(self):
        # Lidar com valores ausentes e inválidos no database.
        # Substituir os valores negativos de idade pela idade média
        self.base_credit.loc[self.base_credit['age'] < 0, 'age'] = 40.92
        # Preencher os valores de idade ausentes com a idade média
        self.base_credit.fillna(self.base_credit['age'].mean(), inplace=True)
        # Dividir em variáveis ​​preditivas (X) e alvo (y)
        self.x_credit = self.base_credit.iloc[:, 1:4].values
        self.y_credit = self.base_credit.iloc[:, 4].values
        # Dimensionar as variáveis ​​preditivas
        self.x_credit = self.scale_credit.fit_transform(self.x_credit)
        
    def show_base_credit(self):
        if self.base_credit is not None:
            print(self.base_credit)
        else:
            print("Base de dados ainda não carregada ou processada. Use `load_data` e `preprocess_data` primeiro.")

    def show_default_counts(self, column):
        #Exibe os valores únicos da coluna e suas respectivas contagens.
        if self.base_credit is not None:
            unique_values, counts = np.unique(self.base_credit[{column}], return_counts=True)
            print(f"Valores únicos e contagens na coluna {column}:")
            for value, count in zip(unique_values, counts):
                print(f"  Valor: {value}, Contagem: {count}")
        else:
            print("Base de dados ainda não carregada. Use `load_data` primeiro.")
    
    def split_data(self, test_size=0.25, random_state=0):
        #divide a base em x teste e treinamento e y teste e treinamento
        self.X_credit_treinamento, self.X_credit_teste, self.Y_credit_treinamento, self.Y_credit_teste = train_test_split(
            self.x_credit, self.y_credit, test_size=test_size, random_state=random_state
        )
        
    def save_data(self, filename='credit.pkl'):
        #salva a variavel em um arquivo .pkl
        with open(filename, mode='wb') as f:
            pickle.dump([self.X_credit_treinamento, self.X_credit_teste, self.Y_credit_treinamento, self.Y_credit_teste], f)
    
    def plot_default_count(self):
        # Grafico para contagem de casos padrão
        sns.countplot(x=self.base_credit['default'])
        plt.show()
        
    def plot_age_histogram(self):
        # Grafico histograma da coluna de idade
        plt.hist(x=self.base_credit['age'])
        plt.show()

    def plot_scatter_matrix(self):
        # Grafico age, income, loan
        grafico = px.scatter_matrix(
            self.base_credit, dimensions=['age', 'income', 'loan'], color='default'
        )
        grafico.show()
    
if __name__ == "__main__":
    #Importar database
    processor = CreditDataProcessor('/Users/felip/PycharmProjects/scripts/Machine/basededados/credit_data.csv')
    #Carregar os dados
    processor.load_data()
    #preprocessar os dados
    processor.preprocess_data()
    #Mostrar os dados preprocessados
    # processor.show_base_credit()
    #Mostrar valores únicos e contagens da coluna 'default'
    # processor.show_default_counts(column="default")
    
    processor.split_data()
    processor.save_data()
    
    # Gera gráficos
    processor.plot_default_count()
    processor.plot_age_histogram()
    processor.plot_scatter_matrix()
        



