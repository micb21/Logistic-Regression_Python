import numpy as np
import matplotlib.pyplot as plt

class regressaologistica():

#Função de inicialização da regressão logística
    def init(self, lr, num_iteracoes, erro_min, delta_iteracao):
        self.lr = lr
        self.num_iteracoes = num_iteracoes
        self.erro_min = erro_min
        self.delta_iteracao = delta_iteracao
        self.parametros = None
        self.coefconst = None

#Definição da função sigmóide
    def sigmoid(self, w):
        try:
            return 1 / (1 + np.exp(-w))
        except ZeroDivisionError:
            print("Impossível dividir por 0.")
            exit()

#Treino do modelo segundo o gradiente descendente
    def fit(self, X, Y):

#Inicialização de parâmetros
        n_samples, n_features = np.shape(X)
        self.parametros = np.zeros(n_features)
        self.coefconst = np.zeros(n_samples)

        erro_antigo = np.zeros(n_samples)

#Método do gradiente descendente
        index_iter=1

        while index_iter < self.num_iteracoes:

#Função sigmóide do modelo linear
            linear_model = np.add(np.dot(X, self.parametros), self.coefconst)
            y_predicted = self.sigmoid(linear_model)

#Atualização dos parâmetros
            d_parametros = (1/ n_samples) * np.dot(np.transpose(X), (y_predicted - Y))
            d_coefconst = (1/ n_samples) * np.sum(y_predicted - Y)

            self.parametros = self.parametros - self.lr * d_parametros
            self.coefconst -= self.lr * d_coefconst

#Cálculo do erro
            erro = np.subtract(y_predicted, Y)

#Condições de paragem
            erro = np.sum(erro)

            if not ((index_iter != 1 and erro_antigo < self.erro_min) and ((erro - erro_antigo) < self.delta_iteracao)):
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
    def plot_2(self, X, Y):
        #plt.plot(X, Y, color='blue', linewidth=3)
        plt.plot(X, self.sigmoid(X), color="blue")
        plt.scatter(X, self.predict(X), color="black")
        plt.show()
        
#Definição da matriz
def criacao_matrix(matrix, linhas, colunas):
    for index_linha in range(linhas):
        print('Insira elementos da linha', index_linha + 1, ' \n')
        linha_atual = []
        for index_coluna in range(colunas):
            linha_atual.append(float(input()))
        matrix.append(linha_atual)
    return np.array(matrix)
