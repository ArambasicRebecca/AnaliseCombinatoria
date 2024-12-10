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

    # Verificar se TOTMES_FINAL e TOTLIQ_FINAL existem no dataset
    if 'TOTMES_FINAL' not in dataset.columns or 'TOTLIQ_FINAL' not in dataset.columns:
        raise ValueError("As colunas 'TOTMES_FINAL' e 'TOTLIQ_FINAL' não foram encontradas no dataset.")

    # Detectar outliers usando o intervalo interquartil (IQR)
    def detectar_outliers(data, coluna):
        Q1 = data[coluna].quantile(0.25)  # Primeiro quartil
        Q3 = data[coluna].quantile(0.75)  # Terceiro quartil
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = data[(data[coluna] < limite_inferior) | (data[coluna] > limite_superior)]
        nao_outliers = data[(data[coluna] >= limite_inferior) & (data[coluna] <= limite_superior)]
        return outliers, nao_outliers

    outliers, nao_outliers = detectar_outliers(dataset, 'TOTMES_FINAL')

    print(f"Número de Outliers em TOTMES_FINAL: {len(outliers)}")
    print(f"Número de Valores Corretos em TOTMES_FINAL: {len(nao_outliers)}")

    # Gráfico de dispersão: Outliers vs Valores Corretos
    plt.figure(figsize=(10, 6))
    plt.scatter(nao_outliers['TOTMES_FINAL'], nao_outliers['TOTLIQ_FINAL'], label="Valores Corretos", alpha=0.7, color='blue')
    plt.scatter(outliers['TOTMES_FINAL'], outliers['TOTLIQ_FINAL'], label="Outliers", alpha=0.7, color='red')
    plt.title("Gráfico de Dispersão - TOTMES_FINAL vs TOTLIQ_FINAL")
    plt.xlabel("TOTMES_FINAL")
    plt.ylabel("TOTLIQ_FINAL")
    plt.legend()
    plt.grid(True)
    plt.show()

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
