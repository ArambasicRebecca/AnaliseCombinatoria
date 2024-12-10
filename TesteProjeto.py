import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Caminho do arquivo
file_path = "C:/AnaliseCombinatoria/Transparencia1224.csv"

try:
    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Erro: O arquivo não foi encontrado no caminho: {file_path}")

    # Carregar o arquivo CSV com o separador correto
    dataset = pd.read_csv(file_path, sep="\t")  # Corrigido para tabulação
    print("Arquivo carregado com sucesso!")
    print("Colunas disponíveis no dataset:")
    print(dataset.columns)

    # Ajustar os nomes das colunas, se necessário
    dataset.columns = dataset.columns.str.strip()  # Remover espaços extras nos nomes

    # Definir X (variáveis preditoras) e Y (alvo)
    X = dataset[['TOTMES_FINAL', 'TOTLIQ_FINAL']]  # Ajuste conforme necessário
    Y = dataset['CARGO']

    # Dividir os dados em treinamento e teste
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Construir o modelo
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train, Y_train)

    # Avaliar o modelo
    accuracy = clf.score(X_test, Y_test)
    print(f"Acurácia do Modelo: {accuracy:.2f}")

    # Visualizar a árvore de decisão
    fig, ax = plt.subplots(figsize=(12, 6))
    tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, ax=ax)
    plt.title("Árvore de Decisão - Classificação de Professores")
    plt.show()

except FileNotFoundError as e:
    print(e)

except KeyError as e:
    print(f"Erro: Uma ou mais colunas não foram encontradas. Detalhes: {e}")

except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
