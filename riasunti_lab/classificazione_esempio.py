# ============================================
# EDA + PREPROCESSING + CLASSIFICATION TEMPLATE
# ============================================

# =====================
# 1. LIBRERIE
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# =====================
# 2. LETTURA CSV
# =====================
df = pd.read_csv(
    "dataset.csv",
    low_memory=False,
    na_values=["NA", "ND", "-", "n/a", ""]
)

# =====================
# 3. ISPEZIONE DATASET
# =====================
print("SHAPE:", df.shape)
print("\nHEAD:")
print(df.head())

print("\nINFO:")
df.info()

print("\nDESCRIBE:")
print(df.describe(include="all"))

print("\nTIPI DI DATO:")
print(df.dtypes)

# =====================
# 4. TARGET ANALYSIS
# =====================
print("\nVALUE COUNTS TARGET:")
print(df["quality_cat"].value_counts())

# =====================
# 5. VALORI NULLI
# =====================
print("\nVALORI NULLI PER COLONNA:")
print(df.isnull().sum())

# =====================
# 6. PULIZIA DATASET
# =====================

# Elimina colonne completamente nulle
df.dropna(axis=1, how="all", inplace=True)

# Elimina righe con target mancante
df.dropna(subset=["quality_cat"], inplace=True)

# Esempio fillna
# df["some_column"].fillna("Unknown", inplace=True)

# =====================
# 7. CONVERSIONI TIPI
# =====================

# Stringa → datetime
# df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Stringa → numerico
# df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")

# =====================
# 8. GRAFICI EDA
# =====================

# Istogrammi (distribuzione feature)
df.hist(figsize=(14, 10))
plt.suptitle("Distribuzione delle feature")
plt.show()

# Boxplot feature vs target
features = ["alcohol", "volatile_acidity", "residual_sugar", "density"]

for f in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="quality_cat", y=f)
    plt.title(f"{f} vs Quality Category")
    plt.show()

# Heatmap valori nulli
sns.heatmap(df.isnull(), cbar=False)
plt.title("Mappa valori mancanti")
plt.show()

# Correlation matrix (solo feature numeriche)
plt.figure(figsize=(10, 8))
corr = df.drop("quality_cat", axis=1).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# =====================
# 9. PREPARAZIONE TARGET
# =====================

# Esempio: raggruppamento classi
df["quality_cat"] = df["quality_cat"].replace({
    "B": "A", "C": "A",
    "F": "E", "G": "E"
})

df["quality_cat"] = df["quality_cat"].replace(
    ["A", "B", "C"], "Low"
)
df["quality_cat"] = df["quality_cat"].replace(
    ["D", "E", "F", "G"], "High"
)

print("\nTARGET DOPO RAGGRUPPAMENTO:")
print(df["quality_cat"].value_counts())

# =====================
# 10. FEATURE / TARGET
# =====================
X = df.drop("quality_cat", axis=1)
y = df["quality_cat"]

# Label encoding (se necessario)
# le = LabelEncoder()
# y = le.fit_transform(y)

# =====================
# 11. TRAIN / TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================
# 12. SCALING
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# 13. MODELLO
# =====================
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =====================
# 14. PREDIZIONI
# =====================
y_pred = model.predict(X_test_scaled)

# =====================
# 15. VALUTAZIONE
# =====================
accuracy = accuracy_score(y_test, y_pred)
print("\nACCURACY:", accuracy)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=sorted(y.unique())
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix")
plt.show()

# =====================
# 16. VISUALIZZAZIONE ALBERO
# =====================
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    max_depth=2,
    filled=True,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())]
)
plt.title("Decision Tree (depth=2)")
plt.show()
