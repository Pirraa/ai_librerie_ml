#leggere csv
#stampare head info e describe
#stampare tipi di dato
#stampare il value counts della colonna target
#stampare valori nulli del df (di default per colonna)

#eliminare colonne tutte nulle
#eliminare righe con valori nulli in colonne specifiche, ad esempio con come subset il target
#filla na con un valore specifico

#conversioni di date in datetime e di numeriche in numeric con coerce

#alle colonne posso applicare
#one hot encoding
#label encoding
#discretizzazione in bin con pd.count

#grafici che posso creare
#sempre: istogrammi per distribuzione di singole variabili df.hist()
#sempre: box plot per distribuzione e outlier legando una feature a un target df.boxplot(data=, x=, y=)
#sempre: posso fare heatmap dei valori nulli sns.heatmap(df.isnull(), cbar=False)
#sempre: matrice di confusione, prima creo crosstab, poi normalizzo e poi grafico con sns.heatmap (oppure uso metodi confusion_matrix e ConfusionMatrixDisplay di sklearn)
#feature e target numerici: scatterplot sns.scatterplot(x,y)
#feature e target numerici: heatmap della matrice di correlazione sns.heatmap(df.corr())


# ============================================================
# EDA e PREPROCESSING GENERICO DI UN DATASET TABELLARE
# ============================================================

# =====================
# 1. LIBRERIE
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# 2. LETTURA DEL DATASET
# ============================================================
# Leggere CSV
df = pd.read_csv("dataset.csv", low_memory=False, na_values=["NA","ND","-","n/a",""])

# Stampare informazioni generali
print("HEAD:")
print(df.head())

print("\nINFO:")
df.info()

print("\nDESCRIBE:")
print(df.describe(include='all'))

print("\nTIPI DI DATO:")
print(df.dtypes)

# Stampare value counts del target
target_col = "target"
print("\nVALUE COUNTS DEL TARGET:")
print(df[target_col].value_counts())

# Stampare valori nulli per colonna
print("\nVALORI NULLI PER COLONNA (%):")
print(df.isnull().mean() * 100)

# ============================================================
# 3. PULIZIA DATASET
# ============================================================

# Eliminare colonne tutte nulle
df.dropna(axis=1, how="all", inplace=True)

# Eliminare righe con valori nulli in colonne specifiche
df.dropna(subset=[target_col], inplace=True)

# Riempire valori NaN con un valore specifico
# df["col1"].fillna("Unknown", inplace=True)

# ============================================================
# 4. CONVERSIONE TIPI
# ============================================================

# Convertire stringhe in datetime
# df["date_col"] = pd.to_datetime(df["date_col"], errors="coerce")

# Convertire stringhe in numerico (float/int)
# df["numeric_col"] = pd.to_numeric(df["numeric_col"], errors="coerce")

# ============================================================
# 5. ENCODING
# ============================================================

# Label encoding per colonne categoriche
le = LabelEncoder()
df["cat_col_encoded"] = le.fit_transform(df["cat_col"].astype(str))

# One hot encoding per colonne categoriche
encoder = OneHotEncoder(sparse_output=False, drop=None)  # drop=None = keeps all the columns
status_reshaped = df_alt['status'].values.reshape(-1, 1)
#fit identifica le categorie uniche, transform  crea array 2d di zeri e uno
status_onehot = encoder.fit_transform(status_reshaped)
# Create a DataFrame containing the on-hot encoded status
status_onehot_df = pd.DataFrame(
    status_onehot,
    columns=[f"status_{cat}" for cat in encoder.categories_[0]],
    index=df_alt.index
)
# Append to df_alt
df_alt = pd.concat([df_alt, status_onehot_df], axis=1)

#più semplice con pandas
cat_features = ['gender','ever_married','work_type','Residence_type']
df_encoded = pd.get_dummies(
    df,
    columns=cat_features,
    drop_first=True
)

# Discretizzazione in bin numerici
# bins = [0, 10, 20, 30]
# labels = [0, 1, 2]
# df["binned_col"] = pd.cut(df["numeric_col"], bins=bins, labels=labels, include_lowest=True)

# ============================================================
# 6. GRAFICI INIZIALI (EDA)
# ============================================================

# Istogrammi per distribuzione di singole feature
df.hist(figsize=(12, 8))
plt.suptitle("Distribuzione feature")
plt.show()

# Boxplot di feature vs target
feature_list = ["feature1", "feature2"]
for f in feature_list:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=target_col, y=f)
    plt.title(f"{f} vs {target_col}")
    plt.show()

# Heatmap dei valori nulli
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Mappa valori mancanti")
plt.show()

# Heatmap della matrice di correlazione (solo numeriche)
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Scatterplot tra feature numeriche e target numerico
# sns.scatterplot(x="feature1", y="feature2", hue=target_col, data=df)

# ============================================================
# 7. OPERAZIONI SUL MODELLO E METRICHE (generiche)
# ============================================================

# Matrice di confusione con crosstab (solo esempio)
# cm_df = pd.crosstab(df[target_col], df["pred_col"], rownames=[target_col], colnames=["predicted"])
# cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0)
# sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")

# Matrice di confusione con sklearn
# cm = confusion_matrix(df[target_col], df["pred_col"])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df[target_col].unique()))
# disp.plot(cmap="Blues")
# plt.show()

# ============================================================
# 8. NOTE
# ============================================================

"""
Questo riassunto contiene le principali operazioni di EDA e preprocessing:

- Lettura e descrizione del dataset
- Pulizia: gestione nulli e duplicati
- Conversione tipi: datetime e numerico
- Encoding: label, one-hot, discretizzazione
- Grafici: istogrammi, boxplot, scatterplot, heatmap valori nulli, heatmap correlazioni
- Matrici di confusione per analisi predizioni

I blocchi grafici e di preprocessing sono indipendenti dal modello ML utilizzato.
"""

