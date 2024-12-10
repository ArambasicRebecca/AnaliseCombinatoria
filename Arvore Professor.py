import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Carregar o dataset
url = "https://raw.githubusercontent.com/leusto/Datasets_Livro_IntMineracaoDados/master/qualidade_servico.csv"
dataset = pd.read_csv(url, sep=";")

# Verificar colunas
print("Colunas disponíveis:", dataset.columns)

# Verificar dimensões do DataFrame
print("Dimensões do dataset:", dataset.shape)

# Codificar colunas categóricas
encoded_columns = pd.get_dummies(dataset[['EP', 'QR', 'LE']], drop_first=True)
X = encoded_columns
y = dataset['R']

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Construir e treinar a árvore de decisão
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Visualizar a árvore de decisão
fig, ax = plt.subplots(figsize=(12, 6))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_, ax=ax)
plt.show()

#Referências

#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

#https://www.rasgoml.com/feature-engineering-tutorials/how-to-generate-feature-importance-plots-from-scikit-learn

#https://medium.com/chinmaygaikwad/feature-importance-and-visualization-of-tree-models-d491e8198b0a