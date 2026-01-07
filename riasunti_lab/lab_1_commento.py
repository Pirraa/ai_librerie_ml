from google.colab import drive
drive.mount('/content/drive')
# @title Set your current working directory where the csv file is located
cwd = 'drive/MyDrive/' # Set your current working directory where the csv file is located



# Import required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# Check if file exists
file_path = cwd + 'es_python/air_quality.csv'
print(file_path)
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The CSV file was not found at the path: {file_path}")



# Loading dataset with read_csv (first 100000 rows in this case)
# low_memory=False impedisce di indovinare i tipi di dato leggendo solo un pezzo del file per risparmio memoria
# na_values specifica quali valori considerare come NaN
df = pd.read_csv(file_path,low_memory=False, na_values=['-', 'NA','ND', 'n/a', ''], nrows=1000000)  # Remove nrows to load the entire dataset
print(f"Original dataset size: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nDescriptive statistics:")#conteggio media deviazione standard min max e quartili
print(df.describe(include='all'))
print("\nDistribution of numerical columns:")
#creo istogramma per ogni colonna numerica del file con grafica lungo 12 e largo 6
df.hist(figsize=(12,6)) # Here the dataframe create a plot using matplot lib. The next rows set how to show the plots
plt.tight_layout() # Better layout
plt.show() # Shows the plot



# Removal of completely empty columns
#axis=1 toglie valori nulli lungo le colonne (0 per le righe)
#how=all per eliminare colonne solo se sono tutte nulle
df = df.dropna(axis=1, how='all')
print("\nPercentage of missing values per column (after dropping empty columns):")
#df.isnull(): Crea un DataFrame booleano (True per NaN, False altrimenti).
#.mean(): Calcola la media per ogni colonna. Poiché True è trattato come 1 e False come 0, la media delle booleane è la proporzione di valori mancant
print(df.isnull().mean() * 100)
print(f"Dataset size (should be the original size): {df.shape}\n\n")
# Remove the rows where status has missing values (opzione subset), con inplace=True modifica il df originale
df.dropna(subset='status', inplace=True)
print(df.isnull().mean() * 100)
print(f"Final dataset size after removing rows where status has missing values: {df.shape}")


#one hot encoding trasforma le variabili categoriche in variabili numeriche binarie
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
# ONE-HOT ENCODING of status
# Create a copy of the dataset for alternative approaches
df_alt = df.copy()
# Apply OneHotEncoder to "status"
#sparse_output=False per ottenere array numpy e non matrice sparsa
#se imposto drop=first elimina la prima colonna per evitare multicollinearità
encoder = OneHotEncoder(sparse_output=False, drop=None)  # drop=None = keeps all the columns
#la libreria richiede che i dati siano in formato 2D (n_samples, n_features)
#df['status'] estrae la colonna status, creando una Series (1D), quindi la trasformo in array 2D con reshape
status_reshaped = df_alt['status'].values.reshape(-1, 1)
#fit identifica le categorie uniche, transform  crea array 2d di zeri e uno
status_onehot = encoder.fit_transform(status_reshaped)
# Create a DataFrame containing the on-hot encoded status
# creo nomi del tipo status_Good, status_Moderate, per ogni categoria restituita dall'encoder con il ciclo for 
#index imposta gli stessi indici di df_alt per allineare i dati
status_onehot_df = pd.DataFrame(
    status_onehot,
    columns=[f"status_{cat}" for cat in encoder.categories_[0]],
    index=df_alt.index
)
# Append to df_alt
#creo unico dataframe unendo df_alt con status_onehot_df lungo le colonne (axis=1)
df_alt = pd.concat([df_alt, status_onehot_df], axis=1)
print("\nExample of one-hot encoding on 'status':")
print(df_alt.head())



# Convertion of categorical coumns into numerical
#per selezionare colonna uso nomedataframe["nomecolonna"]
#replace mappa i valori di una colonna con quelli specificati nel dizionario
df["status"] = df["status"].replace({
    "Hazardous": 5,
    "Very Unhealthy": 4,
    "Unhealthy": 3,
    "Unhealthy for Sensitive Groups": 2,
    "Moderate": 1,
    "Good": 0
})
print(df)


#converte diverse colonne categoriche in numeriche
from sklearn.preprocessing import LabelEncoder  # Automatic with LabelEncoder
# Handling missing values in 'pollutant'
#fillna sostituisce i valori NaN con "Unknown"
df["pollutant"] = df["pollutant"].fillna("Unknown")
#per ogni colonna creo un oggetto LabelEncoder e lo uso per trasformare i valori in numeri interi
le_pollutant = LabelEncoder()
#metodo fit fa la stessa cosa di quello del one hot encoder 
#metodo transform trasforma i valori in numeri interi
#converto prima in stringa per evitare errori con valori NaN
df["pollutant"] = le_pollutant.fit_transform(df["pollutant"].astype(str))
le_county = LabelEncoder()
df["county"] = le_county.fit_transform(df["county"].astype(str))
le_sitename = LabelEncoder()
df["sitename"] = le_sitename.fit_transform(df["sitename"].astype(str))


#per selezionare sottoinsieme di colonne uso df[["col1","col2"]]
#.dtypes restituisce i tipi di dato delle colonne selezionate
print(df[["pollutant", "status", "county", "sitename"]].dtypes)
print("\nDistribution of the target variable 'status':")
#value_counts conta le occorrenze di ogni valore unico nella colonna
print(df["status"].value_counts())


#creo una discretizzazione della colonna aqi
#creo copia dataset
df_alt = df.copy()
# Official (EPA) bins and numeric labels 0...5
#sto crando le classi dei valori numerici cioè gli intervalli di aqi 0-50, 51-100, ecc
bins = [0, 50, 100, 150, 200, 300, 500]
#ciascuna classe avrà un valore numerico da 0 a 5
labels_num = [0, 1, 2, 3, 4, 5]  # 0=Good, ..., 5=Hazardous
# Directly create the numeric column for AQI ranges
#uso elenco classi e etichette create in precedenza per creare la nuova colonna
#right=true crea intervalli chiusi a destra e aperti a sinistra quindi (0,50], (50,100], ecc
#include_lowest=True include il valore più basso del primo intervallo
#astype('Int64') converte la colonna in interi, permettendo valori NaN
#cut funzione di pandas per segmentare e ordinare i valori in intervalli
df_alt['aqi_discretized'] = pd.cut(
    df_alt['aqi'],
    bins=bins,
    labels=labels_num,
    right=True,
    include_lowest=True
).astype('Int64')
#rimuovo righe in cui la nuova colonna creata ha valori NaN
df_alt = df_alt[df_alt['aqi_discretized'].notna()]
# Check
print("\nFirst rows with discretized AQI (official threshold):")
#doppie parentesi per selezionare più colonne
print(df_alt[['aqi', 'aqi_discretized']].head(5))
print("\nCount per bin:")
#value_counts conta le occorrenze di ogni valore unico nella colonna, conto anche i NaN con dropna=False
print(df_alt['aqi_discretized'].value_counts(dropna=False))


#La colonna status è la nostra variabile target (etichetta), ma è stata inserita come testo. La colonna aqi è un valore numerico misurato. 
# Ci si aspetta che l'etichetta status corrisponda sempre alla categoria (aqi_discretized) derivata dal valore aqi misurato.
# Comparison between 'status' and 'aqi_discretized'
#creo una series quindi un array booleano con True se i valori delle due colonne sono uguali
comparison = (df_alt['status'] == df_alt['aqi_discretized'])
#aggiungo colonna al dataframe usando la stessa sintassi per accedere ma a sinistra dell'uguale, a destra assegno i valori
df_alt['match'] = comparison
# Count of matches for each status value
#raggruppa per status, conta quante righe hanno match True o False in ogni gruppo, 
# trasforma il risultato in una tabella con colonne per i valori booleani (con unstack) 
# converte i booleani in interi con astype(int) (True → 1, False → 0). 
# Attenzione: se un gruppo non contiene né True né False, unstack() può produrre NaN e la conversione a intero può fallire.
match_counts = df_alt.groupby('status')['match'].value_counts().unstack().astype(int)
print("\nComparison between 'status' and 'aqi_discretized':")
print(match_counts)



# Comparison plot
#disegna uno scatter plot che mette in relazione l'indice delle righe del DataFrame (asse x) con il valore numerico di status (asse y). 
# L'idea è visualizzare riga per riga se il valore di status coincide con la classe calcolata aqi_discretized.
plt.figure(figsize=(12, 6))
#mapPo i colori in base al valore booleano di 'match': arancione per True (corrispondenza), rosso per False (non corrispondenza)
colors = df_alt['match'].map({True: 'orange', False: 'red'}) # Set dots' color
#disegna i punti usando l'indice del DataFrame come posizione orizzontale e il valore numerico di status come posizione verticale; 
# alpha=0.6 rende i punti parzialmente trasparenti per evidenziare le densità. 
plt.scatter(df_alt.index, df_alt['status'], c=colors, alpha=0.6)
plt.xlabel("Examples (dataset rows)")
plt.ylabel("Status (0=Good ➜ 5=Hazardous)")
plt.title("Comparison between 'status' and 'aqi_discretized'")
plt.show()



# Normalized confusion matrix
import seaborn as sns
#matrice di confusione con come etichetta vera (riga) status e come predetta (colonna) aqi_discretized
#rownames e colnames impostano le etichette degli assi
#Se la cella (Riga 2, Colonna 3) ha un valore di 0.15, 
#significa che il 15% dei casi che avevano status = 2 (Unhealthy for Sensitive Groups) 
#sono stati invece classificati come aqi_discretized = 3 (Unhealthy).
cm = pd.crosstab(df_alt['status'], df_alt['aqi_discretized'], rownames=['status'], colnames=['aqi_bin'])
#normalizzo la matrice dividendo ogni elemento per la somma della riga corrispondente
cm_norm = cm.div(cm.sum(axis=1), axis=0)
plt.figure(figsize=(7, 5))
#disegno la heatmap della matrice di confusione normalizzata (mostra colori in scala di verde)
#annot=True mostra i valori numerici nelle celle, fmt ='.2f' formatta i numeri con 2 decimali
#vmin e vmax impostano il range dei valori per la scala di colori
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=1)
plt.xlabel("Discretized AQI")
plt.ylabel("Status")
plt.title("Normalized confusion matrix")
plt.tight_layout()
plt.show()



# Visualization of Missing Values
print("Count of missing values per column:")
print(df.isnull().sum())


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



# Convert 'pm2.5_avg' to numeric and drop rows with missing 'status' or 'pm2.5_avg'
#converto la colonna in numerica, forzando gli errori a NaN con errors='coerce'
df['pm2.5_avg'] = pd.to_numeric(df['pm2.5_avg'], errors='coerce')
#elimino le righe con valori mancanti in 'status' o 'pm2.5_avg'
df = df.dropna(subset=['status', 'pm2.5_avg'])
# Direct plot of pm2.5_avg vs status
plt.figure(figsize=(8, 5))
#Lo Scatter Plot è lo strumento ideale per visualizzare la relazione tra una variabile numerica continua (pm2.5_avg)
#e una variabile ordinale discreta (status)
#alpha=0.5: Rende i punti semitrasparenti per migliorare la visibilità in aree con alta densità di punti sovrapposti.
sns.scatterplot(x=df['pm2.5_avg'], y=df['status'], alpha=0.5)
plt.title("pm2.5_avg vs Air Quality Status")
plt.xlabel("pm2.5_avg")
plt.ylabel("Status (0=Hazardous ➜ 5=Good)")
plt.tight_layout()
plt.show()


#stessa logica di pm2.5 ma per pm10
#converto la colonna in numerica, forzando gli errori a NaN con errors='coerce'
df['pm10_avg'] = pd.to_numeric(df['pm10_avg'], errors='coerce')
#elimino le righe con valori mancanti in 'status' o 'pm10_avg'
df = df.dropna(subset=['status', 'pm10_avg'])
# Direct plot of pm10_avg vs status
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['pm10_avg'], y=df['status'], alpha=0.5)
plt.title("pm10_avg vs Air Quality Status")
plt.xlabel("pm10_avg")
plt.ylabel("Status (0=Hazardous ➜ 5=Good)")
plt.tight_layout()
plt.show()


#L'obiettivo di questo blocco è visualizzare come la tipologia di inquinante monitorato (pollutant, codificato numericamente) 
#si distribuisce rispetto alla gravità dell'inquinamento (status, codificato ordinalmente).
# elimino le righe con valori mancanti in 'status' o 'pollutant'
df = df.dropna(subset=['status', 'pollutant'])
# Direct plot of pollutant vs status
plt.figure(figsize=(8, 5))
#Lo Scatter Plot qui non mira a mostrare una correlazione lineare (come con pm2.5_avg), 
#ma a visualizzare la distribuzione e la frequenza dello status per ogni tipo di inquinante.
#questo perchè pollutant è una variabile categorica codificata numericamente, non continua
sns.scatterplot(x=df['pollutant'], y=df['status'], alpha=0.5)
plt.title("pollutant vs Air Quality Status")
plt.xlabel("pollutant")
plt.ylabel("Status (0=Hazardous ➜ 5=Good)")
plt.tight_layout()
plt.show()


#### Exercise
#Plot `date`, `hour`, `no2` and `03` vs status.
# Convert date values from strings to datetime objects, which Python and Pandas can interpret as real dates
# Define the features to plot
df['date'] = pd.to_datetime(df['date'], errors='coerce')
#df['date_local'] =df['date_local'].astype('datetime64[ns]')
df['hour']=df['date'].dt.hour
# Drop rows with missing 'status', 'date', 'hour', 'no2', or 'o3'
df = df.dropna(subset=['status', 'date', 'hour', 'no2', 'o3'])
# Plot date vs status
plt.figure(figsize=(12, 6))
#features_to_plot = ['no2','o3','hour']
#sns.scatterplot(data=df, x=feature, y='status', alpha=0.5)
sns.scatterplot(x=df['date'], y=df['status'], alpha=0.5)
plt.title("date vs Air Quality Status")
plt.xlabel("date")
plt.ylabel("Status (0=Hazardous ➜ 5=Good)")
plt.tight_layout()
plt.show()

# Plot hour vs status
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['hour'], y=df['status'], alpha=0.5)
plt.title("hour vs Air Quality Status")
plt.xlabel("hour")
plt.ylabel("Status (0=Hazardous ➜ 5=Good)")
plt.tight_layout()
plt.show()

# Plot no2 vs status
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['no2'],y=df['status'],alpha=0.5)
plt.title("no2 vs Air Quality Status")
plt.xlabel("no2")
plt.ylabel("Status (0=Hazardous ➜ 5=Good)")
plt.tight_layout()
plt.show()



# Correlation matrix on numeric features
#L'obiettivo di questo blocco di codice è misurare e visualizzare la forza e la direzione 
# della relazione lineare (correlazione di Pearson) tra tutte le coppie di variabili numeriche
# nel tuo dataset, con un focus speciale sul target (status).
numeric_df = df.select_dtypes(include=['float64', 'int64'])
import seaborn as sns
#con corr ritorno coefficiente di correlazione di Pearson tra tutte le coppie di colonne numeriche
# se vale 1 significa correlazione positiva perfetta, -1 negativa perfetta, 0 nessuna correlazione
correlation_matrix = numeric_df.corr()  # .corr() returns a square table with Pearson correlation coefficients
# Plot heatmap
plt.figure(figsize=(14, 10))  # Define 14×10 inch figure for clear visualization
#coolwarm mappa i valori di correlazione in una scala di colori che va dal blu (correlazione negativa) al rosso (correlazione positiva)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Environmental Variables vs Status")
plt.xticks(rotation=45)#rotazione etichette asse x di 45 gradi per leggibilità
plt.yticks(rotation=0)
plt.tight_layout()  # Optimize spacing and margins
plt.show()