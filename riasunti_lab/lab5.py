#CLUSTERING CON KMEANS
#rimuov colonne che causano target leakage
#elimino righe con valori mancanti in colonne rimaste
#isolo le feature numeriche e le standardizzo (escludi target)
#standardizza le feature numeriche
#definisci numero cluster, di solito uguale al numero di classi nella colonna target
#addestra il modello (fit_predict) e assegna ogni campione al cluster più vicino
#crea tabella di contingenza tra etichette predette e reali
#mappa ogni cluster al valore di status più frequente in quel cluster
#crea una serie con gli status predetti per ogni campione del dataset
#compara stato predetto con stato reale per ogni riga del dataset (accuratezza globale)
#applico pca per visualizzare i risultati in 2D (coloro punti in base ai cluster predetti)
#calcolo valori medi delle variabili per ogni cluster

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state

#APPLICA ALGORITMO KMEANS AL DATASET LSB5 E VALUTA LA QUALITA' DEL CLUSTERING
#prova tutti i diversi valori di numero di cluster per kmeans per trovare quello ottimale
scores = {}
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled,labels)
    scores[k] = score
    print(f"Silhouette score for k={k}: {score:.4f}")
best_k = max(scores, key=scores.get)

#il numero di cluster è pari al numero di label uniche della colonna target
num_clusters = df['status'].nunique()
# Create a KMeans model with k = num_clusters clusters.
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# Fit the model on the standardized numeric features and assign each sample to its nearest centroid.
labels = kmeans.fit_predict(df_scaled)

#crea una tabella di contingenza con le etichette predette e quelle reali
#dentro df status ci sono i valori reali, dentro labels le etichette predette per ogni riga (campione del dataset) usando le varie features numeriche
#nella tabella le righe sono i valori reali di status, le colonne sono i cluster predetti
#esempio riga status=0, colonna cluster=1, valore=5 significa che 5 campioni con status reale 0 sono stati assegnati al cluster 1
ct = pd.crosstab(df['status'], labels, colnames=['cluster'])
print("Contingency: status vs cluster")
display(ct)

#per ogni colonna(cluster predetto) trovo la riga(status reale) con il valore massimo
#crea un dizionario che mappa ogni cluster al valore di status più frequente in quel cluster
cluster_to_status = ct.idxmax(axis=0).to_dict()
print("Mapping cluster --> status:", cluster_to_status)

#creo una serie con gli status predetti per ogni campione del dataset
#uso indici uguali a quelli di df per poter confrontare facilmente con la colonna reale df['status']
#uso map per mappare un indice numerico (cluster) al valore di status corrispondente usando il dizionario cluster_to_status creato prima
pred_status = pd.Series(labels, index=df.index).map(cluster_to_status)

#compara stato predetto con stato reale per ogni riga el dataset, creo una serie booleana
is_correct = pred_status.eq(df['status'])


#creo un dataframe di riepilogo con 2 colonne (stato e correttezza) e tante righe quante ne ha df
#raggruppo per valore uguale della colonna status le righe
#dopo aver separato per status considera solo la colonna correct
#calcola per ogni gruppo (status) la somma dei valori in correct (cioè il numero di campioni  con colonna correct true) e il conteggio totale dei campioni
#crea due nuove colonne correct (somma) e tot (conteggio)
summary = (
    pd.DataFrame({'status': df['status'], 'correct': is_correct})
      .groupby('status')['correct']
      .agg(correct='sum', tot='count')
)
#aggiungo colonne per incorrect e accuracy
#incorrect sono i campioni totali meno quelli corretti, quindi quelli sbagliati
#accuracy è la percentuale di campioni corretti sul totale
summary['incorrect'] = summary['tot'] - summary['correct']
summary['accuracy_%'] = (summary['correct'] / summary['tot'] * 100).round(2)

#con sort_values indico di ordinare il dataframe summary in base alla colonna accuracy_%
print("\nSummary by status (correct/incorrect/accuracy):")
display(summary.sort_values('accuracy_%', ascending=False))
# calcolo porzione di true nella serie booleana quindi accuratezza globale 
#media sui booleani (True + True + False + True + False) / 5 = (1 + 1 + 0 + 1 + 0) / 5 = 3 / 5 = 0.60
overall_acc = is_correct.mean()
print(f"Global accuracy (cluster --> status): {overall_acc*100:.2f}%")


#VISUALIZZA I RISULTATI DEL CLUSTERING CON PCA 2D
# 2D visualization with PCA
# creo oggetto pca con due dimensioni (grafico che visualizzerò) 
#fit calcola i vettori principali e transform li proietta su 2 direzioni
#df_scaled è il dataframe con solo colonne numeriche e a cui è stato applicato StandardScaler
pca = PCA(n_components=2)
#reduced è un array con due colonne e tante righe quante ne ha df_scaled
#in ogni riga la prima colonna è la componente x e la seconda la componente y
reduced = pca.fit_transform(df_scaled)

# Select the base colormap "tab10"
#è una mappa di 10 coliri distinti per visualizzare categorie discrete
base_cmap = plt.colormaps["tab10"]
# Take only the first 'num_clusters' colors
#estraggo i primi num_clusters colori dalla mappa (colors è lista di tuple rgba)
colors = base_cmap.colors[:num_clusters]
# Create a discrete colormap with these selected colors
#cmap è una lista di colori usata per colorare i punti nel grafico in base al cluster
cmap = ListedColormap(colors)

#preparo label reali per il grafico pca trasformandole in numeri
status_labels = df['status'].replace({'Moderate': 0, 'Good': 1, 'Unhealthy for Sensitive Groups': 2, 'Unhealthy': 3, 'Very Unhealthy': 4}, inplace=False).infer_objects(copy=False)
# Scatter plot of the 2D projection, colored by KMeans cluster labels
plt.figure(figsize=(8,6))#alta 8 pollici e larga 6 pollici
#al metodo scatter passo la coordinata x, la coordinata y, l'array con l'id dei cluster, la color map, la trasparenza, la dimensione dei punti
#predette
sc1 = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap=cmap, alpha=0.6, s=12)
#reali
sc1 = plt.scatter(reduced[:,0], reduced[:,1], c=status_labels, cmap=cmap, alpha=0.6, s=12)

#prelevo id dei cluster una sola volta per creare la legenda
unique_clusters = np.unique(labels)

# Create a colored patch for each cluster
#per ogni cluster assegno un colore e creo lista di colori rgba
colors = [cmap(i) for i in range(num_clusters)]
#creo oggetto grafico per la legenda colorato e con etichetta
#scorro sia per indice che per valore del cluster e assegno colore e label 
patches = [mpatches.Patch(color=colors[i], label=f'Cluster {cl}')
           for i, cl in enumerate(unique_clusters)]
# Add the legend to the plot
#loc è posizione della legenda, frameon disegna bordo attorno alla legenda
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

# Compute cluster sizes to display in the title
#bincut conta quante volte ogni indice appare in labels e restituisce array di occorrenze
sizes = np.bincount(labels)
#creo stringa con formato i:dimensione per ogni cluster
sizes_txt = ", ".join(f"{i}:{sizes[i]}" for i in range(len(sizes)))

# Title with k, variance explained by PCs, and cluster sizes
plt.title(f'KMeans — PCA 2D (k={num_clusters}) | size [{sizes_txt}]')
#aggiusta margini e mostra figura
plt.tight_layout()
plt.show()


#CONFRONTO TRA CLUSTER E VALORI REALI USANDO PCA 2D
# Color points by correctness: 'orange' if cluster --> status mapping matches the true status, else 'red'
#creo array di colori in base alla condizione di correttezza, in questo caso uso una maschera booleana calcolata prima
colors = np.where(is_correct.values, 'orange', 'red')
#  2D PCA scatter colored by match/mismatch
#questa volta prendo sempre reduced come x e y ma coloro i punti in base alla correttezza del mapping cluster-->status
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=colors, alpha=0.6, s=12)
# Legend using colored patches
#questa volta la legenda (patches è il rettangolino della legenda) è più semplice perchè ha solo due categoria, per ognuna indico colore e label
patches = [
    mpatches.Patch(color='orange', label='Match cluster=status'),
    mpatches.Patch(color='red', label='Mismatch')
]
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)
# Title shows overall accuracy computed earlier; axis labels include variance explained by PCs
plt.title(f'PCA comparison — Match (orange) vs Mismatch (red) | Acc {overall_acc*100:.1f}%')
plt.tight_layout()
plt.show()