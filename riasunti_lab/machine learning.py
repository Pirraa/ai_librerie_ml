import pandas as pd
# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()

# Print summary statistics in next line
home_data.describe()
#mostra colonne
melbourne_data.columns
#droppa valori nulli
melbourne_data = melbourne_data.dropna(axis=0)
#seleziono target di predizione (y)
y = melbourne_data.Price
#seleziono features (colonne per prevedere)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

#definisci un modello di decision tree
from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model, cioè cattura pattern dai dati
melbourne_model.fit(X, y)
#effettua la predizione sui valori del dataset 8come esempio, ma dovrei usare valori esterni)
print(melbourne_model.predict(X.head()))

#The prediction error for each house is: error=actual−predicted.With the MAE metric, we take the absolute value of each error. This converts each error to a positive number
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

#exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called validation data.
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#overfitting: albero troppo ampio, tante foglie e pochi esempi in ogni foglia, affidabile su dati di training ma non su validazione (cattura pattern spuri che non ricapitano)
#underfitting: albero poco ampio, pochi gruppi, predizioni sbagliate anche su dati di training (non cattura distinzioni e pattern)
#valore di profondità dell'albero sulla base del minimo di curva di validazione è quello ottimo
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#random forest (molto simile ma crea più alberi e fa la media delle predizioni di ogni albero)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))