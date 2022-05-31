# Etapa 1: Baixar o conjunto de dados em formato CSV (comma-separated-values).

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Etapa 2: Carregar o conjunto de dados
# Ler o dataset tic-tac-toe.csv com separador "," e eliminar a 
data = pd.read_csv('tic-tac-toe.csv', sep=',', header=None, encoding='utf-8')

# Remover a primeira linha
data = data.drop(data.index[0])

# Etapa 3: Converter os valores presentes no conjunto de dados para números inteiros, de acordo com este mapeamento:
data = data.replace(['o', 'b', 'x', 'negativo', 'positivo'], [-1, 0, 1, -1, 1])
data = data.astype(int)
print(data.head())
# print(data.shape)

#  Separar o conjunto de dados em duas partes:
#  80% para treinamento e 20% para testes.
# Etapa 3: Separar o conjunto de dados em duas partes:
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=0)

# calculando a porcentagem de dados do X_train e y_train
print('Porcentagem de dados do X_train: ', round(X_train.shape[0] / data.shape[0] * 100, 1), '%')
print('Porcentagem de dados do X_test: ', round(X_test.shape[0] / data.shape[0] * 100, 1), '%')

# Criar o classificador
NUMBER_ITER = 1000
# Etapa 4: Escolher um dos modelos de classificação a seguir:
classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=NUMBER_ITER, solver='adam', verbose=True, tol=0.00001, random_state=0)

# Treinar o classificador
classifier.fit(X_train, y_train)
print('Treinamento finalizado')

# Etapa 5: Exibir métricas de avaliação, para que possa ser verificada a acurácia do modelo.
# Testar o classificador
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Avaliar o classificador
# Imprimir a acc com duas casas decimais de precisão
print(f'Acurácia final: {acc*100:.2f}%')

# extra
matrix_confusao = confusion_matrix(y_test, y_pred)
print('Matriz de confusão: \n', matrix_confusao)
# extra
print('Relatório de classificação: \n', classification_report(y_test, y_pred))

# Criar gráfico para matriz de confusão
plt.figure(figsize=(10, 10))
plt.imshow(matrix_confusao, interpolation='nearest')
plt.title('Matriz de confusão')
# Setar os valores na matriz de confusão
plt.xticks(range(2), ['O', 'X'])
plt.yticks(range(2), ['O', 'X'])
# Setar o valor da matriz de confusão no gráfico
for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix_confusao[i, j], ha='center', va='center', color='red', fontsize=20)

plt.colorbar()
plt.show()
plt.savefig('matriz_confusao.png')

plt.close()
# Salvar gráfico para matriz de confusão

# save the model to disk
# extra
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
filename = 'finalized_model.pkl'
joblib.dump(classifier, filename)

# Etapa 6: Criar uma opção que permita ao usuário inserir dados arbitrários que devem ser classificados
def verificarVitoriaX(classifierX, X):
    result = classifierX.predict(X)[0]
    print('result: ', result)
    if(result == 1):
        print('Vitória de X: ',  result)
    else: 
        print('Derrota de X: ',  result)


print('\nExemplo de entrada: b,b,o,x,o,x,o,x,x')
entrar_lista = input("Digite a lista de valores (9 valores) (o,b,x): ")
entrar_lista = entrar_lista.replace('o', '-1')
entrar_lista = entrar_lista.replace('x', '1')
entrar_lista = entrar_lista.replace('b', '0')

# Converter para  uma lista de inteiros
entrar_lista = entrar_lista.split(',')
entrar_lista = [[int(i) for i in entrar_lista]]

print('Entrar lista: ', entrar_lista)
verificarVitoriaX(classifier, entrar_lista)