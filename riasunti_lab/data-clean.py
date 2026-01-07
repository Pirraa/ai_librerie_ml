import seaborn as sns
import matplotlib.pyplot as plt
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")

#valori mancanti per ogni colonna, valori mancanti nelle prime 10 colonne
missing_values_count = nfl_data.isnull().sum()
missing_values_count[0:10]

#totale celle, totale valori mancanti e percentualr
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()
# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100

#seleziono una colonna e calcolo lunghzza stringhe in quella colonna, poi conto valori per ogni lunghezza diversa
date_lengths = earthquakes.Date.str.len()
date_lengths.value_counts()
#seleziono indici in cui le date hanno dimensione diversa e stampo le righe relative
indices = np.where([date_lengths == 24])[1]
earthquakes.loc[indices]

#seleziono una colonna dal dataframe e creo dataframe separato
original_data = pd.DataFrame(kickstarters_2017.usd_goal_real)
#ottengo maschera ooleana , cioè array di booleani con true se soddisfo condizione e false se non la soddisfo
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0
#se passo a loc una maschera booleana mi restituisce le righe complete con quegli indici
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

#rimuove colonne con almeno un valore mancante
columns_with_na_dropped = nfl_data.dropna(axis=1)
print("Columns in original dataset: %d \n" % nfl_data.shape[1])

#con loc prendo sottoinsieme di dataset, : per tutte le righe, e le colonne da epa a season
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
#rimpiazza valori nan con 0
subset_nfl_data.fillna(0)
landslides['date'].dtype

y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(X, y)
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)

#converto data in formato comprensibile
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
# multiple date formats in a single column infer_datetime_format=True (pandas indovina il formato)
#ora seleziono giorni mesi anni
day_of_month_landslides = landslides['date_parsed'].dt.day

#decode da byte a formato, encode da formato a byte
before = "This is the euro symbol: €"
after = before.encode("utf-8", errors="replace")
print(after.decode("utf-8"))
sample_entry = b'\xa7A\xa6n'
before = sample_entry.decode("big5-tw")
new_entry = before.encode()

#indovino formato dei dati sulla base delle prime 1000 righe
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result = charset_normalizer.detect(rawdata.read(10000))
#apro on encoding specificato
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')
#salvo dataframe di pandas come csv
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")

#ottengo lista valori di una colonna (utile per uniformare)
countries = professors['Country'].unique()
countries.sort()
# convert to lower case
professors['Country'] = professors['Country'].str.lower()
# remove trailing white spaces
professors['Country'] = professors['Country'].str.strip()

#Fuzzywuzzy returns a ratio given two strings. The closer the ratio is to 100, the smaller the edit distance between the two strings
#controlla quanti caratteri cambiare per trasformare una stringa in un'altra
# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")
replace_matches_in_column(df=professors, column='Country', string_to_match="south korea")


#scaling cambia il range dei dati (rientrano in specifico range come 0-100), utile per confrontare unità di misura diverse
#genero punti casuali da distribuzione esponenziale
original_data = np.random.exponential(size=1000)
# mix-max scale the data between 0 and 1, si può fare con sklearn o con mlxtend
from sklearn.preprocessing import MinMaxScaler
scaled_data = MinMaxScaler().fit_transform(original_data.reshape(-1, 1))
#metto clolumns=0 se ottengo il dataframe da un array np 
from mlxtend.preprocessing import minmax_scaling
scaled_data = minmax_scaling(original_data, columns=[0])
#per scalare su una specifica colonna, se uso pd.Dataframe(nomedf.colonna) ottengo dataframe con come colonna quella selezionata
scaled_data = minmax_scaling(original_data, columns=['usd_goal_real'])

# serie è colonna di dati con valore e indice. faccio[0] per prendere il primo parametri di boxcox
# cn name do nome alla colonna scalata e con index assegno indici
normalized_pledges = pd.Series(stats.boxcox(positive_pledges)[0], name='usd_pledged_real', index=positive_pledges.index)

#normalization cambia la dimensione della distribuzione dei dati
# normalize the exponential data with boxcox (restituisce una tupla, il primo valore è l'array di valori, il secondo il parametro di trasformazione)
#richiede valori positivi
from scipy import stats
normalized_data = stats.boxcox(original_data)

#creo figura di due righe e una colonna, con ax accedo ai due subplot (oggeti axes)
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()

