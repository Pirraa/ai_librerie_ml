# 0. Importazione Librerie (come da fonti)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns

# Inizializzazione del DataFrame (sostituire con il caricamento dati)
df = pd.read_csv('5b_winequality-white_cat.csv')
target = 'quality_cat' # Variabile target

# 1. Caricamento e Pulizia Iniziale
# Rimuovere colonne completamente vuote
df = df.dropna(axis=1, how='all')
# Rimuovere righe con target mancante,
df = df.dropna(subset=[target])
# Sostituire valori infiniti con NaN,
df.replace([np.inf, -np.inf], np.nan, inplace=True)
#bilanciamento
print(df['quality_cat'].value_counts(dropna=False))
low_quality  = ['A', 'B', 'C']
high_quality = ['E', 'F', 'G']
df['quality_cat'] = df['quality_cat'].replace(low_quality, 'Low')
df['quality_cat'] = df['quality_cat'].replace(high_quality, 'High')
df['quality_cat'] = df['quality_cat'].replace('D', 'Medium')
# 2. Identificazione delle Classi e Rimozione Colonne Inutili
num_classes = df[target].nunique() # Numero di classi univoche per il livello di output,
# Colonne non utili per la predizione (ID, testi, target correlati)
#cols_to_drop = ["ID", "colonna_testuale_1", "target_continuo_correlato"]
#df = df.drop(columns=cols_to_drop, errors='ignore')

# 3. Selezione delle Feature (Codifica se necessario)
# (Se sono state mantenute feature categoriche, applicare LabelEncoder qui)
# Esempio: le = LabelEncoder(); df['feat_cat'] = le.fit_transform(df['feat_cat'])
# Selezione delle sole colonne numeriche come feature (X)
feature_cols = df.select_dtypes(include=np.number).columns.tolist()
if target in feature_cols:
    feature_cols.remove(target) # Assicurarsi che il target non sia nelle feature

# 4. Gestione Valori Mancanti (Rimozione righe con NaN nelle feature numeriche)
df = df.dropna(subset=feature_cols, how='any') # Rimuove righe con NaN nelle feature,

X = df[feature_cols].values
y = df[target].values

# 5. Split del Dataset (Train, Validation, Test)
# Primo split: Train/Test (es. 80/20)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify aiuta con classi sbilanciate
)
# Secondo split: Train/Validation (es. 10% del totale = 1/8 del set Train/Val)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val
)

# 6. Codifica del Target (Label Encoding -> One-Hot Encoding)
encoder = LabelEncoder()
encoder.fit(y_train) # Fit solo sul training
y_train_idx = encoder.transform(y_train) # Target come interi (per metriche)
y_val_idx   = encoder.transform(y_val)
y_test_idx  = encoder.transform(y_test)
# One-Hot Encoding per Keras
y_train_oh = to_categorical(y_train_idx, num_classes=num_classes)
y_val_oh   = to_categorical(y_val_idx, num_classes=num_classes)
y_test_oh  = to_categorical(y_test_idx, num_classes=num_classes)

# 7. Standardizzazione delle Feature (Fit solo su Training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit SOLO sul set di Training,
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# 8. Definizione del Modello (Architettura MLP)
input_shape = X_train.shape[1]
model = Sequential([
    Input(shape=(input_shape,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(num_classes, activation="softmax") # Output layer con Softmax,
])

# 9. Compilazione del Modello
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy", # Loss per OHE e Softmax,
    metrics=["accuracy"]
)

# 10. Addestramento
history = model.fit(
    X_train_scaled, y_train_oh,
    validation_data=(X_val_scaled, y_val_oh), # Usa il Validation set,
    epochs=50, # Numero di epoche
    batch_size=128,
    verbose=0

)

# 11. Valutazione e Predizione
loss, acc = model.evaluate(X_test_scaled, y_test_oh, verbose=0)
print(f"Test Accuracy: {acc:.4f}")
# Predizioni (probabilità) e conversione in indici di classe
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred_idx = y_pred_probs.argmax(axis=1) # Conversione in interi (argmax)
print(classification_report(y_test_idx, y_pred_idx, zero_division=0))

# 12. Visualizzazione dei Risultati
class_names = encoder.classes_ # Nomi delle classi originali
# Matrice di Confusione Normalizzata
cm = confusion_matrix(y_test_idx, y_pred_idx, normalize="true") # Matrice normalizzata,
plt.figure(figsize=(10,8))
# Plot heatmap,
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()
# Plot delle curve Loss e Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Loss trend during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
