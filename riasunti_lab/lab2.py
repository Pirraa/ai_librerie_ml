#ALBERI DI CLASSIFICAZIONE CON DECISION TREE CLASSIFIER
#rimuovo colonne vote e righe con target nullo
#seleziono set di feature 
#raggruppo classi poco rappresentate del target in una classe unica
#splitto il dataset in training e test
#standardizzo le feature per portarle sulla stessa scala
#addestro il modello sui dati di training
#faccio predizioni sui dati di test
#valuto il modello con accuracy e classification report
#visualizzo matrice di confusione per analizzare performance
#visualizzo l'albero decisionale con plot_tree

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

#valori null
df = df.dropna(axis=1, how='all')
df.dropna(subset='status', inplace=True)

X = df[features]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # es. 30% per il test
    random_state=42
)


# Scaling con standard e non minmax come kaggle
scaler = StandardScaler()
X_train_scaled_group = scaler.fit_transform(X_train_group)  # fit computes parameters (mean μ and std σ) only on training data
X_test_scaled_group  = scaler.transform(X_test_group)       # apply the μ and σ computed from training set

print("\nTraining set shape:", X_train_group.shape)
print("Testing set shape: ", X_test_group.shape)
print("Shapes after scaling:", X_train_scaled_group.shape, X_test_scaled_group.shape)


#decisiontreeclassifier e non decisiontreeregressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the model
model_group = DecisionTreeClassifier(random_state=42)
model_group.fit(X_train_scaled_group, y_train_group)

# Predictions
y_pred_group = model_group.predict(X_test_scaled_group)

# Evaluation
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test_group, y_pred_group):.2f}")


#tree
print("Features used:", X.columns.tolist())

plt.figure(figsize=(20, 10))
plot_tree(
    model_group,
    max_depth=2,
    feature_names=X.columns,
    class_names=[str(cls) for cls in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=5  # increase the text size in the nodes
)
plt.title("Decision Tree (Depth ≤ 2)")
plt.show()



# Confusion Matrix
cm = confusion_matrix(y_test_group, y_pred_group)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()