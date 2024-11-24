from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import numpy as np


class RedeNeuralCredit:
    def __init__(self, model_path):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self._load_data()
        self._apply_smote()

        # Simplificando a rede neural
        self.rede_neural = MLPClassifier(
            max_iter=300, verbose=True, tol=1e-4,
            solver='adam', activation='tanh', hidden_layer_sizes=(5,),
            alpha=0.01, random_state=42, early_stopping=True,
            learning_rate_init=0.001
        )

    def _load_data(self):
        # Carrega os dados de treinamento e teste do arquivo pickle.
        with open(self.model_path, 'rb') as f:
            self.X_train, self.X_test, self.y_train, self.y_test = pickle.load(f)

        # Aplicando o StandardScaler
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def _apply_smote(self):
        # Aplica o SMOTE para balancear os dados
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def train_model(self):
        # Treina a rede neural com os dados de treinamento.
        print("Treinando o modelo...")
        self.rede_neural.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Avalia o modelo com os dados de teste.
        print("\nAvaliando o modelo...")
        predictions = self.rede_neural.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average="weighted")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")

        print("\nRelatório de Classificação:")
        print(classification_report(self.y_test, predictions))

        print("\nMatriz de Confusão:")
        print(confusion_matrix(self.y_test, predictions))

    def cross_validate(self):
        # Realiza validação cruzada nos dados de treinamento.
        print("\nRealizando validação cruzada...")
        scores = cross_val_score(self.rede_neural, self.X_train, self.y_train, cv=5, scoring="accuracy")
        print(f"Acurácia média (Cross-Validation): {np.mean(scores):.4f}")
        print(f"Desvio padrão (Cross-Validation): {np.std(scores):.4f}")

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
        return np.array(inputs).reshape(1, -1)

    def predict_user_input(self):
        # Faz a previsão com base nos dados fornecidos pelo usuário.
        user_input = self._get_user_input()
        scaled_input = self.scaler.transform(user_input)
        prediction = self.rede_neural.predict(scaled_input)
        print("Previsão para os dados fornecidos:", prediction)


if __name__ == "__main__":
    model_path = '/Users/felip/PycharmProjects/scripts/Machine/credit/credit.pkl'
    credit_nn = RedeNeuralCredit(model_path)

    # Treinar e avaliar o modelo
    credit_nn.train_model()
    credit_nn.evaluate_model()

    # Validação cruzada
    credit_nn.cross_validate()

    # Prever dados fornecidos pelo usuário
    credit_nn.predict_user_input()
