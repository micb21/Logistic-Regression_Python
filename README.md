--- Regressão Logistica ---

Este é o método mais utilizado para modelar variáveis categóricas.
A regressão logística é o equivalente da regressão linear, mas aplicada a problemas de classificação. Este tipo de problema surge quando queremos categorizar alguma variável por classes. 

Para implementar, são efetuadas duas modificações ao modelo linear. Ao modelar uma variável binária, queremos saber a probabilidade dela ser 0 ou 1, utilizando para isso a função sigmóide. Além disso, é regularizada pela função custo com fatores exponenciais que penalizam o modelo em caso de erros na predição, evitando o overfitting.

--- Exemplo de Aplicação ---

regressao = regressaologistica()

lr = 0.001
delta_iteracao = 0.001

num_iteracoes = float(input('Insira o número maximo de iteracoes: '))

erro_min = float(input('Erro mínimo para condição de paragem: '))

regressao.init(lr, num_iteracoes, erro_min, delta_iteracao)

l_x = int(input('Insira número de linhas/colunas Matriz X: '))

c_x=l_x

X = []

X = criacao_matrix(X, l_x, c_x)

#l_y = int(input('Insira numero de linhas/colunas Matriz Y: '))

c_y=l_x

l_y=c_y

Y = []

Y = criacao_matrix(Y, l_y, c_y)

#X = np.array([[3, 2], [1, 2]])

#Y = np.array([[4, 2], [3, 2]])

regressao.fit(X,Y)

regressao.plot_2(X,Y)

![alt text](https://github.com/micb21/RegressaoLogistica/blob/main/Figure_1.png?raw=true)
