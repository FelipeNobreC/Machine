from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from census.SVM import X_census_treinamento
from credit.BaseCreditPP import x_credit

with open('/Users/felip/PycharmProjects/MA e Data/credit/credit.pkl', 'rb') as f:
  X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)

# 3 -> 100 -> 100 -> 1
# 3 -> 2 -> 2 -> 1
rede_neural_credit = MLPClassifier(max_iter=1500, verbose=True, tol=0.0000100,
                                   solver = 'adam', activation = 'relu',
                                   hidden_layer_sizes = (20,20))
rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)


previsoes = rede_neural_credit.predict(X_credit_teste)
print(previsoes)
print(X_credit_teste)

print(accuracy_score(y_credit_teste, previsoes))

#print(classification_report(y_credit_teste, previsoes))

def trata_dados_credit():
    scale_credit = StandardScaler()

    # Ajusta o escalonador com os dados de treinamento (não é necessário treinar toda vez que você chama esta função)
    scale_credit.fit(X_credit_treinamento)  # Ajusta o escalonador aos dados de treinamento

    # Variáveis para armazenar as entradas do usuário
    variavel = []

    # Recebe os valores como números (não como strings)
    while True:
        try:
            valor = int(input('Digite o valor do income: '))
            break  # Sai do loop se o valor for válido
        except ValueError:
            print("Por favor, insira um número válido.")
    variavel.append(valor)  # Adiciona income

    while True:
        try:
            valor = int(input('Digite o valor do age: '))
            break  # Sai do loop se o valor for válido
        except ValueError:
            print("Por favor, insira um número válido.")
    variavel.append(valor)  # Adiciona age

    while True:
        try:
            valor = int(input('Digite o valor do loan: '))
            break  # Sai do loop se o valor for válido
        except ValueError:
            print("Por favor, insira um número válido.")
    variavel.append(valor)  # Adiciona loan

    # Converte a lista de variáveis para uma matriz 2D (1 amostra, 3 características)
    variavel = np.array(variavel).reshape(1, -1)  # Agora 'variavel' tem 1 amostra e 3 características

    # Normaliza os dados com o scaler já ajustado aos dados de treinamento
    variavel2 = scale_credit.transform(variavel)  # Aplica a normalização com os dados de treinamento

    return variavel2



def predict_rede_n():
    # Obtemos os valores tratados com a função trata_dados_credit
    valores_tratados = trata_dados_credit()  # Aqui você já tem os dados tratados e normalizados

    # Criamos e treinamos a rede neural
    rede_neural_credit = MLPClassifier(max_iter=1500, verbose=True, tol=0.0000100,
                                       solver='adam', activation='relu',
                                       hidden_layer_sizes=(20, 20))

    # A rede já foi treinada com os dados X_credit_treinamento, então você só precisa de um modelo treinado
    rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)  # Treinando a rede

    # Previsão usando os dados tratados como entrada para a rede neural
    previsoes = rede_neural_credit.predict(valores_tratados)  # Passando os dados tratados para a previsão
    print("Previsões:", previsoes)  # Imprime as previsões

predict_rede_n()

