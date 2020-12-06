--- Regressão Logistica ---

Este é o método mais utilizado para modelar variáveis categóricas.
A regressão logística é o equivalente da regressão linear, mas aplicada a problemas de classificação. Este tipo de problema surge quando queremos categorizar alguma variável por classes. 

Para implementar, são efetuadas duas modificações ao modelo linear. Ao modelar uma variável binária, queremos saber a probabilidade dela ser 0 ou 1, utilizando para isso a função sigmóide. Além disso, é regularizada pela função custo com fatores exponenciais que penalizam o modelo em caso de erros na predição, evitando o overfitting.

--- Exemplo de Aplicação ---

regressao = regressaologistica()

lr = 0.001
delta_iteracao = 0.001

print('Insira o numero maximo de iteracoes \n')
num_iteracoes = float(input())

print('Erro minimo para condicao de paragem \n')
erro_min = float(input())

regressao._init_(lr, num_iteracoes, erro_min, delta_iteracao)

print('Insira numero de linhas Matriz X \n')
l_x = int(input())
print('Insira numero de colunas Matriz X \n')
c_x = int(input())
X = []
X = criacao_matrix(X, l_x, c_x)

print('Insira numero de linhas Matriz Y \n')
l_y = int(input())
print('Insira numero de colunas Matriz Y \n')
c_y = int(input())
Y = []
Y = criacao_matrix(Y, l_y, c_y)
X = [[0.3, 0.8], [0.1, 0.2]]
Y = [[0.5, 0.3], [0.3, 0.5]]

regressao.fit(X,Y)
regressao.plot(X,Y)