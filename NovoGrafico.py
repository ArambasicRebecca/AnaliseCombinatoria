import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Configuração para reprodutibilidade
tf.random.set_seed(1)

# Gerar dados de exemplo
X = np.random.rand(2000, 15, 1)  # Aumentado para 2000 amostras, 15 timesteps
Y = np.random.randint(0, 2, 2000)  # Saídas binárias (0 ou 1)

# Normalização dos dados
scaler = MinMaxScaler()
X = X.reshape(-1, 1)  
X = scaler.fit_transform(X)
X = X.reshape(2000, 15, 1)

# Divisão dos dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criação do modelo com melhorias
model = Sequential([
    Input(shape=(15, 1)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(32),  
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Configuração de Early Stopping e Model Checkpoint
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Treinamento do modelo
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop, checkpoint]
)

# Avaliação do modelo
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Perda no conjunto de teste: {test_loss:.4f}")
print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")

# Visualizar o histórico de treinamento com mais métricas
plt.figure(figsize=(12, 5))

# Gráfico da Perda
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda - Treinamento')
plt.plot(history.history['val_loss'], label='Perda - Validação')
plt.title('Evolução da Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Gráfico da Acurácia
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acurácia - Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia - Validação')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()

# Carregar o melhor modelo salvo
best_model = tf.keras.models.load_model('best_model.keras')
print("Melhor modelo carregado com sucesso!")

# Previsões no conjunto de teste
Y_pred_prob = best_model.predict(X_test)  # Probabilidades preditas
Y_pred = (Y_pred_prob > 0.5).astype(int)   # Convertendo probabilidades para classes

# Matriz de Confusão
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.show()

# Curva ROC
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Linha diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Histograma das Previsões
plt.figure()
plt.hist(Y_pred_prob[Y_test == 1], bins=20, alpha=0.5, label='Classe Positiva (1)', color='blue')
plt.hist(Y_pred_prob[Y_test == 0], bins=20, alpha=0.5, label='Classe Negativa (0)', color='orange')
plt.title('Distribuição das Previsões')
plt.xlabel('Probabilidade Predita')
plt.ylabel('Frequência')
plt.legend()
plt.show()