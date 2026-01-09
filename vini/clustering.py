import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Per la standardizzazione,
from sklearn.cluster import KMeans             # Per l'algoritmo di clustering,
from sklearn.decomposition import PCA          # Per la riduzione della dimensionalità,
from sklearn.metrics import silhouette_score   # Opzionale: per scegliere K,
from matplotlib.colors import ListedColormap   # Per la colorazione dei cluster nei plot,
import matplotlib.patches as mpatches

# 1. Pulizia Iniziale e Selezione Feature
#Caricamento dati
df = pd.read_csv('5b_winequality-white_cat.csv')
# Rimuovere colonne completamente vuote
df = df.dropna(axis=1, how='all')
# Rimuovere righe con target di riferimento mancante, mostrare quanti e quali target esistono
target_reference = 'quality_cat'
df.dropna(subset=[target_reference], inplace=True)
print(df[target_reference].unique())
print("Different values:", df[target_reference].nunique())
# Colonne da rimuovere per evitare target leakage e non numeriche (come suggerito)
#cols_to_drop = ["sitename", "county", 'aqi', "siteid", "pollutant", "date"]
#df = df.drop(columns=cols_to_drop, errors='ignore')

# 2. Isolamento Feature Numeriche e Gestione Valori Mancanti
# Selezione delle sole colonne numeriche rimanenti
df_numeric = df.select_dtypes(include=[np.number])
# KMeans non gestisce NaN, li elimino
df_final = df.dropna(subset=df_numeric.columns, how="any")
# Aggiornamento delle feature e del target dopo la pulizia finale
X = df_final[df_numeric.columns].values # Feature set (solo numeriche)
y_ref = df_final[target_reference].values # Target di riferimento (solo per comparazione)

# 3. Standardizzazione delle Feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Standardizzazione cruciale per KMeans,

# 4. Definizione del Numero di Cluster (K)
# K impostato sul numero di classi uniche del target di riferimento (per comparazione)
num_clusters = df_final[target_reference].nunique() # Ad esempio: 5 classi,
# [Opzionale: calcolare il silhouette score per scegliere K ottimale se non si usa un riferimento esterno]

# 5. Addestramento del Modello
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X_scaled) # Etichette di cluster assegnate

# 6. Analisi e Mappatura (Creazione Tabella di Contingenza)
# Aggiungere le etichette di cluster al DataFrame per l'analisi
df_final['Cluster'] = labels
# Tabella di contingenza
ct = pd.crosstab(df_final[target_reference], df_final['Cluster'], colnames=['cluster'])
display(ct)
# Mappatura Cluster -> Status (Regola del voto di maggioranza),
cluster_to_status = ct.idxmax(axis=0).to_dict()
pred_status = df_final['Cluster'].map(cluster_to_status)

# 7. Valutazione (Comparativa)
# Calcolo accuratezza globale (media delle previsioni corrette)
is_correct = pred_status.eq(df_final[target_reference])
overall_acc = is_correct.mean()
print(f"Accuratezza Globale (cluster --> status): {overall_acc*100:.2f}%")
# Calcolo accuratezza per classe (usando la mappatura)
# [Il codice qui richiede l'aggregazione dei risultati come mostrato nelle fonti

# 8. Visualizzazione: Riduzione Dimensionalità (PCA)
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled) # Dati proiettati su 2 componenti principali
base_cmap = plt.colormaps["tab10"]
colors = base_cmap.colors[:num_clusters]
cmap = ListedColormap(colors)
# Plot dei cluster in 2D predetti(colorati in base all'etichetta del cluster)
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap=cmap, alpha=0.6, s=12)
plt.title(f'KMeans — PCA 2D (k={num_clusters})')
plt.tight_layout()
plt.show()
# codice aggiuntivo legenda
#unique_clusters = np.unique(labels)
#colors = [cmap(i) for i in range(num_clusters)]
#patches = [mpatches.Patch(color=colors[i], label=f'Cluster {cl}')
           #for i, cl in enumerate(unique_clusters)]
#plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)
#sizes = np.bincount(labels)
#sizes_txt = ", ".join(f"{i}:{sizes[i]}" for i in range(len(sizes)))
#plt.title(f'KMeans — PCA 2D (k={num_clusters}) | size [{sizes_txt}]')

#plot reale (non predetto)
# Mapping 'A' to 0 (lowest quality) and 'G' to 6 (highest quality)
quality_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
status_labels = df['quality_cat'].replace(quality_mapping).infer_objects(copy=False)
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=status_labels, cmap=cmap, alpha=0.6, s=12)
plt.title(f'Labels distribution')
plt.tight_layout()
plt.show()
#codice aggiuntivo legenda
#colors = [cmap(i) for i in range(num_clusters)]
#patches = [mpatches.Patch(color=colors[i], label=cl)
            #for i, cl in enumerate(df['status'].unique())]
# Add the legend to the plot
#plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

#plot confronto cluster e valori reali
colors = np.where(is_correct.values, 'orange', 'red')
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=colors, alpha=0.6, s=12)
plt.title(f'PCA comparison — Match (orange) vs Mismatch (red) | Acc {overall_acc*100:.1f}%')
plt.tight_layout()
plt.show()
#codice aggiuntivo legenda
#patches = [
    #mpatches.Patch(color='orange', label='Match cluster=status'),
    #mpatches.Patch(color='red', label='Mismatch')
#]

# 9. Interpretazione: Calcolo dei Valori Medi per Cluster
cluster_summary = df_final.drop(columns=['Cluster']).groupby(labels).mean(numeric_only=True)
display(cluster_summary) # Mostra il profilo medio delle feature per ogni cluster