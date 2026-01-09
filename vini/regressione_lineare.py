# 0. Importazione Librerie
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. caricamento e statistiche iniziali
df = pd.read_csv('5b_winequality-white_cat.csv')
print("SHAPE:", df.shape)
print("\nHEAD:")
print(df.head())
print("\nINFO:")
df.info()
print("\nDESCRIBE:")
print(df.describe(include="all"))
print("\nTIPI DI DATO:")
print(df.dtypes)
print("\nVALORI NULLI PER COLONNA:")
print(df.isnull().sum())

#2. Grafici
#grafici esplorativi isorgrammi delle feature
df.hist(figsize=(14, 10))
plt.suptitle("Distribuzione delle feature")
plt.show()
# Heatmap valori nulli
sns.heatmap(df.isnull(), cbar=False)
plt.title("Mappa valori mancanti")
plt.show()
# Correlation matrix (solo feature numeriche)
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Matrice di correlazione")
plt.show()
# Scatter plot di alcune feature rispetto al target 'alcohol'
important_features = ["density", "residual_sugar", "fixed_acidity"]
target_col = 'alcohol'
for f in important_features:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[f], df[target_col], alpha=0.6)
    plt.xlabel(f)
    plt.ylabel(target_col)
    plt.title(f"{f} vs {target_col}")
    plt.show()

#3.pre elaborazioni varie
# Rimuovere colonne completamente vuote (es. 'unit')
df = df.dropna(axis=1, how='all')
# Codifica delle colonne categoriche in numerico (ad esempio 'county' e 'sitename')
categorical_cols_to_encode = ['quality_cat']
for col in categorical_cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
# Selezionare solo le colonne numeriche come feature (escludendo il target 'aqi')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target_col)
# Rimuovere righe con valori mancanti nel target
df_clean = df.dropna(subset=numeric_cols+[target_col]).copy()
# Allineare X e y
X = df_clean[numeric_cols]
y = df_clean[target_col]


# 4. Split del Dataset (Train/Test) [9]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # Percentuale per il test set
    random_state=42 # Per riproducibilità
)

# 5. Addestramento del Modello
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predizione
y_pred = model.predict(X_test)

# 7. Valutazione (MSE e RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)  # squared=False restituisce RMSE
rmse_mine = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"RMSE on aqi range (%): {(rmse/(y_test.max()-y_test.min()))*100:.2f}")

# 8. Visualizzazione (Scatter Plot)
plt.figure(figsize=(8, 6))
# Scatter plot dei valori reali (Actual) vs valori predetti (Predicted)
plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted values')
# Linea di predizione perfetta (y=x)
min_val = y_test.min()
max_val = y_test.max()
plt.plot([min_val, max_val], [min_val, max_val], color='red', label='Perfect prediction')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Linear Regression: Actual AQI vs Predicted AQI")
plt.legend()
plt.show()

#regressione polinomiale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# Define a polynomial model of degree 2
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
