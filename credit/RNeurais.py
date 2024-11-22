from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler


class RedeNeuralCredit:
    def __init__(self, model_path):
        self.model_path = model_path
        self.scaler = StandardScaler()
        # 3 -> 100 -> 100 -> 1
        # 3 -> 2 -> 2 -> 1
        self.rede_neural = MLPClassifier(
            max_iter=1500, verbose=True, tol=0.0000100,
            solver='adam', activation='relu', hidden_layer_sizes=(20, 20)
        )
        self._load_data()
    
    def _load_data(self):
        # Carrega os dados de treinamento e teste do arquivo pickle.
        with open(self.model_path, 'rb') as f:
            self.X_train, self.X_test, self.y_train, self.y_test = pickle.load(f)
        self.scaler.fit(self.X_train)  # Ajusta o escalonador aos dados de treinamento
        
    def train_model(self):
        # Treina a rede neural com os dados de treinamento.
        print("Treinando o modelo...")
        self.rede_neural.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        # Avalia o modelo com os dados de teste.
        predictions = self.rede_neural.predict(self.X_test)
        print(predictions)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Accuracy: {accuracy}")
        print("Relatório de Classificação:")
        print(classification_report(self.y_test, predictions))

    def _get_user_input(self):
        # Solicita ao usuário os valores de entrada (income, age, loan).
        inputs = []
        for feature in ['income', 'age', 'loan']:
            while True:
                try:
                    value = int(input(f"Digite o valor de {feature}: "))
                    inputs.append(value)
                    break
                except ValueError:
                    print("Por favor, insira um número válido.")
        return np.array(inputs).reshape(1, -1) # Retorna um array NumPy de uma linha que terá uma coluna para cada valor em inputs

    def predict_user_input(self):
        # Faz a previsão com base nos dados fornecidos pelo usuário.
        user_input = self._get_user_input() # Pega os inputs
        scaled_input = self.scaler.transform(user_input) # Aplica a normalização com os dados de inputs
        prediction = self.rede_neural.predict(scaled_input) # Previssao do modelo de rede neural
        print("Previsão para os dados fornecidos:", prediction)

    def generic(self):
        pass

if __name__ == "__main__":
    model_path = '/Users/felip/PycharmProjects/scripts/Machine/credit/credit.pkl'
    credit_nn = RedeNeuralCredit(model_path)

    # Treinar e avaliar o modelo
    credit_nn.train_model()
    credit_nn.evaluate_model()

    # Prever dados fornecidos pelo usuário
    credit_nn.predict_user_input()


