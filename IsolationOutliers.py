import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Gerar dados de exemplo
np.random.seed(0)
data = np.random.normal(loc=50, scale=10, size=1000)
data_with_outliers = np.concatenate([data, [120, 130, 140]])  # Adicionando outliers

# Criar DataFrame
df = pd.DataFrame(data_with_outliers, columns=['Value'])

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Value'])
plt.title('Boxplot para Identificação de Outliers')
plt.show()

# Z-Score
scaler = StandardScaler()
df['Z-Score'] = scaler.fit_transform(df[['Value']])
outliers_z = df[np.abs(df['Z-Score']) > 3]
print("Outliers identificados pelo Z-Score:")
print(outliers_z)

# Histogramas
plt.figure(figsize=(12, 6))
plt.hist(df['Value'], bins=30, alpha=0.7, color='blue')
plt.axvline(x=outliers_z['Value'].mean(), color='red', linestyle='dashed', linewidth=1)
plt.title('Histograma com Outliers')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.show()

# PCA para identificação de outliers
pca = PCA(n_components=2)
data_pca = pca.fit_transform(df[['Value']])
plt.scatter(data_pca[:, 0], np.zeros_like(data_pca[:, 0]), alpha=0.5)
plt.title('PCA - Visualização dos Dados')
plt.xlabel('Componente Principal 1')
plt.show()

# Isolation Forest
iso_forest = IsolationForest(contamination=0.01)  # Ajuste o parâmetro conforme necessário
df['Outlier'] = iso_forest.fit_predict(df[['Value']])
outliers_if = df[df['Outlier'] == -1]
print("Outliers identificados pelo Isolation Forest:")
print(outliers_if)