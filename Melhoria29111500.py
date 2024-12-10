import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Configuração para reprodutibilidade
tf.random.set_seed(1)

# Gerar dados de exemplo
X = np.random.rand(2000, 15, 1)  # Aumentado para 2000 amostras, 15 timesteps
Y = np.random.randint(0, 2, 2000)  # Saídas binárias (0 ou 1)

# Normalização dos dados
scaler = MinMaxScaler()
X = X.reshape(-1, 1)  # Ajustar para aplicar o scaler
X = scaler.fit_transform(X)
X = X.reshape(2000, 15, 1)  # Retorna ao formato original: (samples, timesteps, features)

# Divisão dos dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criação do modelo com melhorias
model = Sequential([
    Input(shape=(15, 1)),  # Especifica o formato da entrada: (timesteps, features)
    LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),  # Dropout para evitar overfitting
    BatchNormalization(),  # Normalização para estabilizar o aprendizado
    LSTM(64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu'),  # Camada totalmente conectada intermediária
    Dense(1, activation='sigmoid')  # Camada de saída para classificação binária
])

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Configuração de Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# Configuração de Model Checkpoint para salvar o melhor modelo
checkpoint = ModelCheckpoint(
    'best_model.keras',  # Use '.keras' para formato mais recente ou '.h5' com save_format
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Treinamento do modelo
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop, checkpoint]  # Incluindo early stopping e checkpoint
)

# Avaliação do modelo
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

# Carregar o melhor modelo salvo
best_model = tf.keras.models.load_model('best_model.keras')
print("Melhor modelo carregado com sucesso!")
