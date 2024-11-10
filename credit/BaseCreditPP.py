import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle


#abre o csv
base_credit = pd.read_csv('/Users/felip/PycharmProjects/MA e Data/basededados/credit_data.csv')
#localiza na coluna age do csv onde tem valores menores que 0 e subistitue pela media
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92
#filtra na coluna age valores sem preencher e coloca no lugar a media da coluna
base_credit.fillna(base_credit['age'].mean(), inplace = True)

#separa a base em 2, uma para os valores preditivos sendo a x e a y para a previsao respondida
x_credit = base_credit.iloc[:,1:4].values
y_credit = base_credit.iloc[:,4].values

#armazena a classe standardscaler
scale_credit = StandardScaler()
# usa a classe standardscaler para ajustar os valores na mesma escala
x_credit = scale_credit.fit_transform(x_credit)
#print(base_credit)
#print(np.unique(base_credit['default'], return_counts= True))
def grafic_default(): # gera e abre janela do grafico
    sns.countplot(x=base_credit['default'])
    plt.show()

#divide a base em x teste e treinamento e y teste e treinamento
X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste = train_test_split(x_credit, y_credit, test_size=0.25, random_state=0)

#salva a variavel em um arquivo .pkl
with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste], f)


#graficos
plt.hist(x = base_credit['age'])
#plt.show()

""""base_credit2 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
variavel que recebe a basecredit.drop(passando a basecredit[na posicao age
e filtrando as que sao menores que  0, assim pegamos o indice com o .index e passamos para o
.drop a lista com os indices para ele apagar
print(base_credit2)"""

grafico = px.scatter_matrix(base_credit, dimensions=['age','income','loan'], color = 'default' )
#grafico.show()
#print(base_credit.loc[base_credit['age'] < 0])
#print(base_credit.head(33))
#print(a)
#print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])
#print(x_credit)


