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

    # Carregar o arquivo CSV
    dataset = pd.read_csv(file_path, sep="\t")  # Ajuste o separador conforme necessário
    print("Arquivo carregado com sucesso!")
    print(dataset.head())
    print("Colunas disponíveis no dataset:")
    print(dataset.columns)

    # Verificar e ajustar tipos de dados das colunas relevantes
    if 'NOME' in dataset.columns and 'REFERENCIA' in dataset.columns:
        dataset['NOME'] = dataset['NOME'].astype(str)
        dataset['REFERENCIA'] = dataset['REFERENCIA'].astype(str)

        # Ordenar o dataset
        dataset = dataset.sort_values(by=['NOME', 'REFERENCIA'])
        print("Dataset ordenado com sucesso!")

    else:
        raise ValueError("As colunas 'NOME' e 'REFERENCIA' não estão no dataset.")

    # Verificar se TOTMES_FINAL e TOTLIQ_FINAL existem no dataset
    if 'TOTMES_FINAL' not in dataset.columns or 'TOTLIQ_FINAL' not in dataset.columns:
        raise ValueError("As colunas 'TOTMES_FINAL' e 'TOTLIQ_FINAL' não foram encontradas no dataset.")

    # Definir X (variáveis preditoras) e Y (alvo)
    X = dataset[['TOTMES_FINAL', 'TOTLIQ_FINAL']]
    Y = dataset['CARGO']

    # Dividir os dados em treinamento e teste
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Construir o modelo de árvore de decisão
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train, Y_train)

    # Avaliar o modelo
    accuracy = clf.score(X_test, Y_test)
    print(f"Acurácia do Modelo: {accuracy:.2f}")

    # Visualizar a árvore de decisão
    fig, ax = plt.subplots(figsize=(12, 6))
    tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, ax=ax)
    plt.title("Árvore de Decisão - Classificação de Cargos")
    plt.show()

except FileNotFoundError as e:
    print(e)

except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
