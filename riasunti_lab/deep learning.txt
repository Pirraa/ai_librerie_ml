#creo rete neurale
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(1, input_shape=[1]), #solo il primo layer di solito ha input_shape cioè le variabili x (features) da cui dipende la y (intercetta) (colonne del dataset)
    layers.Dense(512,activation='relu',input_shape=[8]) #layer può avere funzione di attivazione e più unità cioè neuroni (512)
    layers.Dense(1) # output layer, nel caso di classificazione metto activation='sigmoid'
])

#se voglio mettere altri layer fra un layer e la funzione di attivazione posso separare le due cose
layers.Dense(units=32, input_shape=[8]),
layers.Activation('relu'),
layers.Dropout(rate=0.3), # apply 30% dropout to the next layer (rimuovo input ai layer per ogni step di training, in modo che non impari pattern spuri)
#batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation, 
#and then also putting the data on a new scale with two trainable rescaling parameters
layers.BatchNormalization(),

#aggiungo funzioni di perdita e di addestramento
model.compile(
    optimizer='adam', #funzione che indica come cambiare i pesi e i bias (usa anche sgd)
    loss='mae', #funzione ch calcola differenza fra valori reali e valori predetti 
    #loss='binary_crossentropy' metrics=['binary_accuracy'], per i problemi di classificazione e x
)

#stoppa de vado in overfitting (curva d validazione cresce e non diminuisce allontanandosi da quella di training)
#If there hasn't been at least an improvement of 0.001 in the validation loss over the previous 20 epochs, then stop the training and keep the best model you found
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

#splitto il dataset in training e validation
spotify = pd.read_csv('../input/dl-course-data/spotify.csv')
X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])
X_train, X_valid, y_train, y_valid = group_split(X, y, artists)
#oppure uso X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)


#creo storico delle funzioni di perdita e inizio addestramento
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256, #256 righe alla volta 
    epochs=10, #ripeti il processo per 10 volte
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

#per plottare la perdita di training e di validazione(fit mantiene in oggetto history il vettore delle loss in ogni epoca)
#se validaton_loss diminuisce e non si allontana da loss allora non ho overfitting
import pandas as pd
# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df.loc[5:, ['loss']].plot(); #da epoca 5 in poi
history_df['loss'].plot();
history_df.loc[0:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot() #per classificazione binaria
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));

#funzioni varie per grafici con matplotlib
x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)
plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()

