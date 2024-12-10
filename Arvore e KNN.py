from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#carregar dados
data = load_iris()
X,Y = data.data, data.target

#dividir os dados em treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X,Y,train_size=0.3, random_state=42)

#criar e treinar o modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_treino, Y_treino)

#fazer previsões
Y_pred = modelo.predict(X_teste)

#avaliar o modelo
print("Acurácia da Árvore de Decisão:", accuracy_score(Y_teste,Y_pred))

#KNN
from sklearn.neighbors import KNeighborsClassifier

#riar e treinar o modelo
knn = KNeighborsClassifier(n_neighbors=3) #usar 3 vizinhos
knn.fit(X_treino,Y_treino)

#Fazer previsões
Y_pred_knn = knn.predict(X_teste)

#Avaliar o modelo
print("Acurácia do KNN: ", accuracy_score(Y_teste,Y_pred_knn))