import joblib

# load the model from disk
filename = 'finalized_model.pkl'
classifier = joblib.load(filename)


entrar_lista = input("Digite a lista de valores (9 valores) (o,b,x): ")
entrar_lista = entrar_lista.replace('o', '-1')
entrar_lista = entrar_lista.replace('x', '1')
entrar_lista = entrar_lista.replace('b', '0')

# Converter para  uma lista de inteiros
entrar_lista = entrar_lista.split(',')
entrar_lista = [[int(i) for i in entrar_lista]]


entrada = [[ 1, 1, 1,
            -1, 1,-1,
             1, -1,-1]]
# print('Entrar entrada: ', entrada)
print('Entrar lista: ', entrar_lista)

def verificarVitoriaX(classifierX, X):
    result = classifierX.predict(X)[0]
    print('result: ', result)
    if(result == 1):
        print('Vitória de X: ',  result)
    else: 
        print('Derrota de X: ',  result)

# verificarVitoriaX(classifier, entrada)
verificarVitoriaX(classifier, entrar_lista)
