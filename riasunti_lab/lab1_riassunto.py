#LIBRERIE VARIE
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder


#IMPORT E STATISTICHE VARIE DATASET
# Loading dataset with read_csv (first 100000 rows in this case)
# low_memory=False impedisce di indovinare i tipi di dato leggendo solo un pezzo del file per risparmio memoria
# na_values specifica quali valori considerare come NaN
df = pd.read_csv(file_path,low_memory=False, na_values=['-', 'NA','ND', 'n/a', ''], nrows=1000000)  # Remove nrows to load the entire dataset
print(f"Original dataset size: {df.shape}")
print(df.head())
print(df.info())
print(df.describe(include='all'))#conteggio media deviazione standard min max e quartili


#MANIPOLARE RIGHE E COLONNE DATASET
#per selezionare sottoinsieme di colonne uso df[["col1","col2"]]
#.dtypes restituisce i tipi di dato delle colonne selezionate
print(df[["pollutant", "status", "county", "sitename"]].dtypes)
#value_counts conta le occorrenze di ogni valore unico nella colonna
print(df["status"].value_counts())
#media valori nulli per colonna
print(df.isnull().mean() * 100)
#conteggio valori nulli per colonna
print(df.isnull().sum())

#ELIMINARE COLNNE NULLE O RIGHE CON VALORI NULI
#axis=1 toglie valori nulli lungo le colonne (0 per le righe)
#how=all per eliminare colonne solo se sono tutte nulle
df = df.dropna(axis=1, how='all')
#eliminare righe con valori nulli in colonne specifiche, inplace=True per modificare il dataframe originale
df.dropna(subset=['status', 'pm10_avg'], inplace=True)
#fillna sostituisce i valori NaN con "Unknown"
df["pollutant"] = df["pollutant"].fillna("Unknown")
#rimuovo righe in cui la nuova colonna creata ha valori NaN
df_alt = df_alt[df_alt['aqi_discretized'].notna()]

#CONVERSIONE TIPI DI DATI
#raggruppa per status, conta quante righe hanno match True o False in ogni gruppo, 
# trasforma il risultato in una tabella con colonne per i valori booleani (con unstack) 
# converte i booleani in interi con astype(int) (True → 1, False → 0). 
# Attenzione: se un gruppo non contiene né True né False, unstack() può produrre NaN e la conversione a intero può fallire.
match_counts = df_alt.groupby('status')['match'].value_counts().unstack().astype(int)
# Convert date values from strings to datetime objects
df['date'] = pd.to_datetime(df['date'], errors='coerce')
#seleziono solo le colonne numeriche del dataframe
numeric_df = df.select_dtypes(include=['float64', 'int64'])
#estrae e ridimensiona una colonna
#df['status'] estrae la colonna status, creando una Series (1D), quindi la trasformo in array 2D con reshape
status_reshaped = df['status'].values.reshape(-1, 1)
#converto la colonna in numerica, forzando gli errori a NaN con errors='coerce'
df['pm10_avg'] = pd.to_numeric(df['pm10_avg'], errors='coerce')


#AGGIUNGERE COLONNE
#aggiunro colonna specificando con dizionario come crearla(trasformo ad esempio categorici in numerali)
df["status"] = df["status"].replace({
    "Hazardous": 5,
    "Very Unhealthy": 4,
    "Unhealthy": 3,
    "Unhealthy for Sensitive Groups": 2,
    "Moderate": 1,
    "Good": 0
})

#COPIA E CONCATENAZIONE DATASET
#creo copia dataset
df_alt = df.copy()
#creo nuovo dataset con colonne specificate da columns (in questo caso ciclo su una lista di nomi di colonne)
status_onehot_df = pd.DataFrame(
    status_onehot,
    columns=[f"status_{cat}" for cat in encoder.categories_[0]],
    index=df_alt.index
)
#concateno due dataset (axis=1 per colonne, axis=0 per righe)
df_alt = pd.concat([df_alt, status_onehot_df], axis=1)

#ONE HOT ENCODING
#sparse_output=False per ottenere array numpy e non matrice sparsa
#se imposto drop=first elimina la prima colonna per evitare multicollinearità
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

#LABEL ENCODIING
#per ogni colonna creo un oggetto LabelEncoder e lo uso per trasformare i valori in numeri interi
le_pollutant = LabelEncoder()
#metodo fit fa la stessa cosa di quello del one hot encoder 
#metodo transform trasforma i valori in numeri interi
#converto prima in stringa per evitare errori con valori NaN
df["pollutant"] = le_pollutant.fit_transform(df["pollutant"].astype(str))

#DISCRETIZZAZIONE COLONNA
#sto crando le classi dei valori numerici cioè gli intervalli di aqi 0-50, 51-100, ecc
bins = [0, 50, 100, 150, 200, 300, 500]
#ciascuna classe avrà un valore numerico da 0 a 5
labels_num = [0, 1, 2, 3, 4, 5]  # 0=Good, ..., 5=Hazardous
df_alt['aqi_discretized'] = pd.cut(
    df_alt['aqi'],
    bins=bins,
    labels=labels_num,
    right=True,
    include_lowest=True
).astype('Int64')


#GRAFICI
#creo istogramma per ogni colonna numerica del file con grafica lungo 12 e largo 6
df.hist(figsize=(12,6)) # Here the dataframe create a plot using matplot lib. 

#SCATTERPLOT
#Lo Scatter Plot è lo strumento ideale per visualizzare la relazione tra una variabile numerica continua (pm2.5_avg)
#e una variabile ordinale discreta (status)
#alpha=0.5: Rende i punti semitrasparenti per migliorare la visibilità in aree con alta densità di punti sovrapposti.
sns.scatterplot(x=df['pm2.5_avg'], y=df['status'], alpha=0.5)
#disegna i punti usando l'indice del DataFrame come posizione orizzontale e il valore numerico di status come posizione verticale; 
# alpha=0.6 rende i punti parzialmente trasparenti per evidenziare le densità. 
colors = df_alt['match'].map({True: 'orange', False: 'red'}) # Set dots' color
plt.scatter(df_alt.index, df_alt['status'], c=colors, alpha=0.6)


#HEATMAP
#con corr ritorno coefficiente di correlazione di Pearson tra tutte le coppie di colonne numeriche
# se vale 1 significa correlazione positiva perfetta, -1 negativa perfetta, 0 nessuna correlazione
correlation_matrix = numeric_df.corr()  # .corr() returns a square table with Pearson correlation coefficients
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# Heatmap of missing values
plt.figure(figsize=(12, 6))
#questa heatmap rappresenta solo true e false, dove true indica la presenza di un valore mancante (NaN) in quel punto del DataFrame
#quindi con cbar=false non mostro la barra dei colori e uso la mappa 'viridis' per i colori
#Se vedi una striscia verticale di un colore diverso (il colore dei valori mancanti) lungo una colonna, significa che quella colonna ha molti valori mancanti.
#L'asse Y rappresenta le singole righe (osservazioni) del dataset.
#L'asse X rappresenta le colonne (variabili) del dataset.
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Map')
plt.show()


#MATRICE DI CONFUSIONE
#matrice di confusione con come etichetta vera (riga) status e come predetta (colonna) aqi_discretized
#rownames e colnames impostano le etichette degli assi
#Se la cella (Riga 2, Colonna 3) ha un valore di 0.15, 
#significa che il 15% dei casi che avevano status = 2 (Unhealthy for Sensitive Groups) 
#sono stati invece classificati come aqi_discretized = 3 (Unhealthy).
#disegno la heatmap (mostra colori in scala di verde)
#annot=True mostra i valori numerici nelle celle, fmt ='.2f' formatta i numeri con 2 decimali
#vmin e vmax impostano il range dei valori per la scala di colori
cm = pd.crosstab(df_alt['status'], df_alt['aqi_discretized'], rownames=['status'], colnames=['aqi_bin'])
cm_norm = cm.div(cm.sum(axis=1), axis=0)#normalizzo dividendo elemento per la somma della riga stessa
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=1)
#disegno la heatmap della matrice di confusione normalizzata (mostra colori in scala di verde)
#annot=True mostra i valori numerici nelle celle, fmt ='.2f' formatta i numeri con 2 decimali
#vmin e vmax impostano il range dei valori per la scala di colori
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=1)

#funzioni generali
plt.figure(figsize=(8,5))
plt.tight_layout() # Better layout
plt.show() # Shows the plot
plt.xlabel("Examples (dataset rows)")
plt.ylabel("Status (0=Good ➜ 5=Hazardous)")
plt.title("Comparison between 'status' and 'aqi_discretized'")
plt.xticks(rotation=45)#rotazione etichette asse x di 45 gradi per leggibilità
plt.yticks(rotation=0)