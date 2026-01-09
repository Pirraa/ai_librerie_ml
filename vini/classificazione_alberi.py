# 0. Importazione Librerie
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # Per lo split del dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder # Per standardizzazione e encoding
from sklearn.tree import DecisionTreeClassifier, plot_tree # Per il modello e la visualizzazione
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, ConfusionMatrixDisplay # Per le metriche

# 1. Caricamento visualizzazione statistiche iniziali
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

#2. grafici esplorativi isorgrammi delle feature e boxplot
df.hist(figsize=(14, 10))
plt.suptitle("Distribuzione delle feature")
plt.show()
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

#3. preelaborazioni e feature engineering
# Conversione data in datetime e estrazione feature temporali (se usate)
# df['date'] = pd.to_datetime(df['date'], errors='coerce')
# df['year'] = df['date'].dt.year
# Codifica delle Feature Categoriche
#categorical_features = ['county', 'sitename'] # Selezionare le feature categoriche da usare
#for col in categorical_features:
    #le = LabelEncoder()
    #df[col] = le.fit_transform(df[col].astype(str))
# Rimuove colonne completamente vuote
df = df.dropna(axis=1, how='all')
target_col = 'quality_cat'
# Rimuove righe con target mancante
df.dropna(subset=[target_col], inplace=True)
# Gestione dei valori mancanti nelle Feature (Dropping rows)
# Definizione delle Feature (X)
features = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol'] # Lista finale delle colonne feature
df = df.dropna(subset=features, how='any')

# 4. Preparazione Target (Opzionale: Bilanciamento/Raggruppamento)
# Esempio: Raggruppamento delle classi meno frequenti (se necessario)
print(df['quality_cat'].value_counts(dropna=False))
low_quality  = ['A', 'B', 'C']
high_quality = ['E', 'F', 'G']
df['quality_cat'] = df['quality_cat'].replace(low_quality, 'Low')
df['quality_cat'] = df['quality_cat'].replace(high_quality, 'High')
df['quality_cat'] = df['quality_cat'].replace('D', 'Medium')

X = df[features]
y = df[target_col]

# 5. Split del Dataset (Train/Test) [19]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # es. 30% per il test
    random_state=42,
    stratify=y
)

# 6. Standardizzazione delle Feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit SOLO sul Training
X_test_scaled = scaler.transform(X_test)       # Transform su Test

# 7. Addestramento del Modello
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Predizione e Valutazione
y_pred = model.predict(X_test_scaled)

# Calcolo Accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Matrice di Confusione e Report
labels = sorted(y_train.unique())  # Usa le etichette dal training set
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues') # Per visualizzare la matrice
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred)) # Opzionale: per metriche dettagliate

# 9. Visualizzazione dell'Albero (Opzionale, con limitazioni)
plt.figure(figsize=(15, 10))
plot_tree(model, max_depth=2, feature_names=X.columns.tolist(), class_names=[str(cls) for cls in labels])
plt.title("Decision Tree Visualization (Depth 2)")
plt.show()
