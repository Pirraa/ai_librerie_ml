#REGRESSIONE LINEARE MULTIPLA
#definisco il target 
#rimuovo colonne interanmente vuote
#codifico colonne categoriche a numeriche con labelencoder
#seleziono solo colonne numeriche come feature
#divido in train e test set
#addestro modello di regressione lineare multipla
#predico valori su test set
#calcolo MSE  RMSE e errore percentuale sul range di valori
#cred scatterplot di valori reali vs predetti con retta y=x

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#seleziona tutte le colonne numeriche del dataset, utile per regressione
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#rimuovo la y da predirre
numeric_cols.remove('aqi')

#elimino righe con valori nulli in una di quelle colonne
df_clean = df.dropna(subset=numeric_cols + ['aqi']).copy()

X = df_clean[numeric_cols.columns]

# Create and train the model
model = LinearRegression()
#fit addestra modello
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)

# Metriche, le ultime due coincidono
#6.66 significa che con range di 196 sto sbagliando in media di 6 (circa 3%)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
rmse_mine = np.sqrt(mse)

#errore percentuale si calcola con: (rmse/range di valori)*100
print(f"RMSE on aqi range (%): {(rmse/(y_test.max()-y_test.min()))*100:.2f}")

# Plot comparison of predicted vs actual values
#asse x valori reali e asse y valori predetti
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted values')
#retta y=x cioè linea di predizione perfetta, diagonale a 45 gradi pendenza 1
#se punto cade sulla linea significa che valore predetto=valore reale
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Perfect prediction')

#regressione polinomiale, in questo caso grado due (il resto è uguale)
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)