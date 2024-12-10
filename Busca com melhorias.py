import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Configurar a semente para aleatoriedade determinística
tf.random.set_seed(1)

# Gerar dados de exemplo
X = np.random.rand(100, 5, 1)  # (samples, timesteps, features)
Y = np.random.randint(0, 2, 100)  # Binário (0 ou 1)

# Dividir os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criar o modelo
model = Sequential([
    Input(shape=(5, 1)),  # Especifica o formato da entrada: (timesteps, features)
    SimpleRNN(50, activation='relu'),  # Camada RNN com 50 neurônios
    Dense(1, activation='sigmoid')  # Camada de saída para classificação binária
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))

# Avaliar o modelo
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Perda no conjunto de teste: {test_loss:.4f}")
print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")

# Visualizar o histórico de treinamento
# Gráfico da Perda
plt.figure()
plt.plot(history.history['loss'], label='Perda - Treinamento')
plt.plot(history.history['val_loss'], label='Perda - Validação')
plt.title('Evolução da Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Gráfico da Acurácia
plt.figure()
plt.plot(history.history['accuracy'], label='Acurácia - Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia - Validação')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
