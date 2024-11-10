import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np


class Modelo():
    def __init__(self):
        self.df = None
        self.model_svc = None
        self.model_lr = None
        self.X_test_svc = None
        self.y_test_svc = None
        self.X_test_lr = None
        self.y_test_lr = None

    def CarregarDataset(self, path):

        #Carrega o conjunto de dados a partir do dataset
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):

        #Realiza o pré-processamento dos dados
        # Visualizar as primeiras linhas do dataset
        print("Primeiras linhas do dataset:")
        print(self.df.head())

        # Verificar a presença de valores ausentes e remover, se houver
        if self.df.isnull().sum().any():
            self.df.dropna(inplace=True)

        # Codificar a coluna 'Species' em valores numéricos para o modelo
        label_encoder = LabelEncoder()
        self.df['Species'] = label_encoder.fit_transform(self.df['Species'])

    def Treinamento(self):

        #Treina os modelos de machine learning
        # Separar variáveis independentes e dependentes para classificação
        X = self.df.drop(columns='Species')
        y = self.df['Species']

        # Dividir o dataset em conjuntos de treino e teste para SVM
        X_train_svc, self.X_test_svc, y_train_svc, self.y_test_svc = train_test_split(X, y, test_size=0.3,
                                                                                      random_state=42)

        # Instanciar e treinar o modelo SVC com validação cruzada
        self.model_svc = SVC()
        cv_scores_svc = cross_val_score(self.model_svc, X_train_svc, y_train_svc, cv=5)
        print("Acurácia média do modelo SVC com validação cruzada:", cv_scores_svc.mean())

        # Treinar o modelo SVC
        self.model_svc.fit(X_train_svc, y_train_svc)

        # Dividir o dataset em conjuntos de treino e teste para Regressão Linear
        X_train_lr, self.X_test_lr, y_train_lr, self.y_test_lr = train_test_split(X, y, test_size=0.3, random_state=42)

        # Instanciar e treinar o modelo de Regressão Linear com validação cruzada
        self.model_lr = LinearRegression()
        cv_scores_lr = cross_val_score(self.model_lr, X_train_lr, y_train_lr, cv=5, scoring='neg_mean_squared_error')
        print("Erro quadrático médio negativo médio do modelo Linear Regression com validação cruzada:",
              -cv_scores_lr.mean())

        # Treinar o modelo de Regressão Linear
        self.model_lr.fit(X_train_lr, y_train_lr)

    def Teste(self):

        #Avalia o desempenho dos modelos treinados nos dados de teste
        # Avaliar o modelo SVC
        y_pred_svc = self.model_svc.predict(self.X_test_svc)
        accuracy_svc = accuracy_score(self.y_test_svc, y_pred_svc)
        print("Acurácia do modelo SVC:", accuracy_svc)

        # Avaliar o modelo de Regressão Linear
        y_pred_lr = self.model_lr.predict(self.X_test_lr)
        mse_lr = mean_squared_error(self.y_test_lr, y_pred_lr)
        print("Erro quadrático médio do modelo Linear Regression nos dados de teste:", mse_lr)

    def Train(self):

        #Função principal para o fluxo de treinamento do modelo
        self.CarregarDataset(file_path)
        self.TratamentoDeDados()
        self.Treinamento()


# Caminho para o arquivo de dados
file_path = 'iris.data'

# Instanciando a classe Modelo e executando o treinamento e teste
modelo = Modelo()
modelo.Train()
modelo.Teste()
