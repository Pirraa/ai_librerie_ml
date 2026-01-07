#RETI NEURALI CON KERAS PER CLASSIFICAZIONE MULTI-CLASSE
#identifico target
#elimino colonne con target nullo
#identifico numero di classi(numero neuroni ultimo livello rete)
#rimuovo colonne non numeriche inutili per la rete neurale
#seleziono colonne numeriche e creo lista
#elimino righe con valori nulli in quelle colonne
#primo split del dataframe in train e test
#secondo split del train in train e val
#prendo da questi dataset 3 array per le feature e 3 array per i target
#label encoder sui  target con classe (fit su train e transform su tutti)
#one hot encoding sui target con to_categorical
#effettuo anche standardizzazione sulle feature con StandardScaler su tutte
#definisco rete con keras.Sequential
#compilo rete con optimizer,loss e metrics
#traino la rete con fit e ottengo history
#evaluation con dati di test x e y
#predizione y con dati x di test
#creo matrice di confusione con dati di test e predetti
#uso sns.heatmap per plottare e matplotlib per mostrare grafico
#plot di loss val_loss accurac e val_accuracy con diverse label

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score


import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical

#individuo quante clasi ci sono nella feature da prevedere
class_names = df['status'].unique()
num_classes=len(df['status'].unique())

#rimuovo colonne non necessarie messe in array (id, unità di misura,text fields)
df = df.drop(columns=columns_to_drop, errors='ignore')

#sostituisco valori infiniti con NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#conto quanti valori ci sono per ogni classe nella feature target dopo la pulizia
print("\nClass count in the cleaned DataFrame:")
print(df[target].value_counts().sort_index())

#Train → Training
#Validation → Scegli i parametri early stopping architettura  migliori il modello
#Test → Giudizio finale, veramente “non visto”

# First split: train/test
train_df, test_df = train_test_split(
    df, # Here we do not create X and y like in previous notebooks, just to see a differrent workflow
    test_size=0.2,
    random_state=42
) #train 80% e test 20%

# Second split: within the training set, we also separate the validation set
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42
)# train 72% (0.8*0.9) , val 8% (0.8*0.1)

#controllo se ci sono classi mancanti in train e test rispetto al dataset completo
all_classes = set(np.unique(df[target]))
train_classes = set(np.unique(train_df[target]))
test_classes  = set(np.unique(test_df[target]))
missing_train = all_classes - train_classes
missing_test  = all_classes - test_classes

# Convert the DataFrames into NumPy arrays (stessa cosa con le y per tutti e 3 i gruppi train,test e validation)
X_train = train_df[feature_cols].values

#✔️ fit + transform separati
#→ quando hai train / val / test e vuoi evitare leakage

#✔️ fit_transform insieme
#→ quando trasformi tutto il dataset prima di dividerlo
#→ oppure quando ti serve un encoding una tantum senza vincoli ML
encoder = LabelEncoder()
encoder.fit(y_train)
y_train_idx = encoder.transform(y_train)
y_val_idx   = encoder.transform(y_val)
y_test_idx  = encoder.transform(y_test)

#dopo il labelencoding applico il one hot encoding ma non uso OneHotEncoding
#uso la funzione to_categorical di keras per convertire gli indici in vettori one hot
y_train_oh = to_categorical(y_train_idx, num_classes=num_classes)
y_val_oh   = to_categorical(y_val_idx,   num_classes=num_classes)
y_test_oh = to_categorical(y_test_idx, num_classes=num_classes)

#definisco la rete neurale
# Define the model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),#layer input con il numero di feature
    layers.Dense(16, activation="relu"),#primo layer hidden con 16 neuroni e funzione di attivazione relu
    layers.Dense(8, activation="relu"),
    layers.Dense(num_classes, activation="softmax")#layer di classificazione con softmax per output multi-classe, crea distribuzione di probabilità
])#per classificazione binaria userei sigmoid e 1 neurone nel layer di output

# Model filling
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",#funzione di perdita per classificazione multi-classe
    #binary_crossentropy per classificazione binaria
    metrics=["accuracy"]
)

# Model training
history = model.fit(
    X_train, y_train_oh,#dataset du training
    validation_data=(X_val, y_val_oh),#se ho fatto secondo split metto anche validation data
    epochs=50,#quante volyte faccio vedere tutto il dataset
    batch_size=128,#quante istanze alla volta, ogni quanto aggiorno i pesi
    verbose=1
)


#definisco le metriche di valutazione
# Model evaluation on the test set
#evaluate prende dataset di test con etichette e calcola accuratezza (predict e confronto automatico)
loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# Predictions on the test set
#come per algoritmi precedenti (classificazione) predico le etichette per il test set
#successivamente confronto con y_test_idx per calcolare confusion matrix
#argmax trasforma outputdi probabilità in etichette originali con onehot encoding
#l'output di model.predict sono le probabilità per ogni classe
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

# Confusion Matrix con valori di test e predetti
cm = confusion_matrix(y_test_idx, y_pred, normalize="true")
plt.figure(figsize=(10,8))
#posso usare sns oppure confusionmatrixdisplay
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)

#durante il training ottengo history che tiene traccia di loss e accuracy per training e validation
#posso fare dei plot per visualizzare l'andamento di queste metriche
#val loss calcolata alla fine di ogni epoca sul validation set
#se val_loss cresce sto facendo overfitting, lì mi fermo e non devo fare più epoche
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')