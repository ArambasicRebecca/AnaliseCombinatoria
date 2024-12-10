import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Configurar a semente para aleatoriedade determinística
tf.random.set_seed(1)

# Gerar dados de exemplo
X = np.random.rand(1000, 10, 1)  # Aumentado para 1000 amostras, 10 timesteps
Y = np.random.randint(0, 2, 1000)  # Saídas binárias (0 ou 1)

# Normalização dos dados
scaler = MinMaxScaler()
X = X.reshape(-1, 1)  # Ajustar para aplicar o scaler
X = scaler.fit_transform(X)
X = X.reshape(1000, 10, 1)  # Retorna ao formato original: (samples, timesteps, features)

# Divisão dos dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criar o modelo otimizado
model = Sequential([
    Input(shape=(10, 1)),  # Especifica o formato da entrada: (timesteps, features)
    LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),  # Dropout para prevenir overfitting
    LSTM(32, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.2),  # Mais uma camada de Dropout
    Dense(1, activation='sigmoid')  # Camada de saída para classificação binária
])

# Compilar o modelo com otimizador RMSprop
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Configurar Early Stopping para evitar overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinar o modelo
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop]
)

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
