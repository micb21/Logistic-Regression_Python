#Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt

class regressaologistica():

#Função de inicialização da regressão logística
    def _init_(self, lr, num_iteracoes, erro_min, delta_iteracao):
        self.lr = lr
        self.num_iteracoes = num_iteracoes
        self.erro_min = erro_min
        self.delta_iteracao = delta_iteracao
        self.parametros = None
        self.coefconst = None

#Definição da função sigmóide
    def sigmoid(self, w):
        return 1 / (1 + np.exp(-w))
    #Nota: 1 dividido por 0 não é possível.

#Treino do modelo segundo o gradiente descendente
    def fit(self, X, Y):

#Inicialização de parâmetros
        n_samples, n_features = np.shape(X)
        self.parametros = np.zeros(n_features)
        self.coefconst = np.zeros(n_samples)

        erro_antigo = np.zeros(n_samples)

#Método do gradiente descendente
        index_iter=1
        should_stop=False

        while index_iter < self.num_iteracoes or should_stop != True:

#Função sigmóide do modelo linear
            linear_model = np.add(np.dot(X, self.parametros), self.coefconst)
            y_predicted = self.sigmoid(linear_model)

#Atualização dos parâmetros
            d_parametros = (1/ n_samples) * np.dot(np.transpose(X), (y_predicted - Y))
            d_coefconst = (1/ n_samples) * np.sum(y_predicted - Y)

            self.parametros = self.parametros - self.lr * d_parametros
            self.coefconst -= self.lr * d_coefconst

# Cálculo do erro
            erro = np.subtract(y_predicted, Y)

#Condições de paragem
            erro = np.sum(erro)

            if (index_iter != 1 and erro_antigo < self.erro_min) and ((erro - erro_antigo) < self.delta_iteracao):
                should_stop=True

            else:
                should_stop=False
                erro_antigo = erro

            index_iter += 1

#Função de previsão do modelo
    def predict(self, X):
        linear_model = np.dot(X, self.parametros) + self.coefconst
        y_predicted = self.sigmoid(linear_model)
        cls_matrix_size = (len(y_predicted), len(y_predicted))
        y_prediction_cls = np.zeros(cls_matrix_size)

        for i in range(len(y_predicted)):
            for j in range(len(y_predicted[i])):
                if y_predicted[i][j] > 0.5:
                    y_prediction_cls[i][j] = 1
                else:
                    y_prediction_cls[i][j] = 0

        return y_prediction_cls

#Gráfico da regressão logística
    def plot(self, X, Y):
        fig = plt.figure()
        plt.plot(X, Y, color='blue', linewidth=3)
        plt.scatter(X, self.predict(X))
        plt.show()

#Definição de matriz na interface com o utilizador
    def criacao_matrix(matrix, linhas, colunas):
        for index_linha in range(linhas):
            print('Insira elementos da linha', index_linha + 1, ' \n')
            linha_atual = []
        for index_coluna in range(colunas):
            linha_atual.append(float(input()))
        matrix.append(linha_atual)
        return matrix