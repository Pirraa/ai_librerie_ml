#This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
# The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset.
# Several constraints were placed on the selection of these instances from a larger database. 
# In particular, all patients here are females at least 21 years old of Pima Indian heritage.2 From the data set in the (.csv) File We can find several variables, 
# some of them are independent (several medical predictor variables) and only one target dependent variable (Outcome).

#classificazione su colonna Outcome con reti neurali
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense,Input
from keras.models import Sequential
from keras.utils import to_categorical

df=pd.read_csv("diabetes.csv")
print("PRIME RIGHE")
print(df.head())
print("\n\nTIPI DI DATO")
print(df.dtypes)
print("\n\nSTATISTICHE")
print(df.describe())
print("\n\nVALORI NULLI")
print(df.isnull().sum())
print("\n\nSHAPE")
print(df.shape)
print("\n\nCOLONNE")
print(df.columns)
print("\n\nINFO")
print(df.info())
target="Outcome"

#grafici esplorativi

#valori nulli
df=df.dropna(axis=1,how='all')
df.replace([np.inf],np.nan,inplace=True)
df.dropna(subset=target,inplace=True)

#encoding bilanciamento target e eliminazione colonne inutili
print("\n\nVALORI TARGET")
print(df[target].value_counts())
num_classes=df[target].nunique()
print(num_classes)

#selezione valori numerici e creazione variabili x e y
df_num_cols=df.select_dtypes(include=np.number).columns.tolist()
df_num_cols.remove(target)
print(df_num_cols)
x=df[df_num_cols].values
y=df[target].values

#split
x_train_val,x_test,y_tran_val,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train,x_val,y_train,y_val=train_test_split(x_train_val,y_tran_val,test_size=0.1)

#label e one hot encoding target
le=LabelEncoder()
y_train_idx=le.fit_transform(y_train)
y_val_idx=le.transform(y_val)
y_test_idx=le.transform(y_test)
y_train_oh=to_categorical(y_train_idx)
y_test_oh=to_categorical(y_test_idx)
y_val_oh=to_categorical(y_val_idx)

#scaling feature
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
x_val_scaled=scaler.transform(x_val)

#creazione modello
model=Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(16,activation='relu'),
    Dense(8,activation='relu'),
    Dense(num_classes,activation='sigmoid')
])

#compile
model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')

#fit e addestrament
history=model.fit(x_train_scaled,y_train_oh,validation_data=(x_val_scaled,y_val_oh),epochs=50,batch_size=128,verbose=0)

#evaluate e metriche
loss,acc=model.evaluate(x_test_scaled,y_test_oh)
print(f'Accuracy: {acc}')
print(f'Loss: {loss}')
y_pred_prob=model.predict(x_test_scaled)
y_pred_idx=y_pred_prob.argmax(axis=1)
print(classification_report(y_test,y_pred_idx))

#matrice di confusione
cm=confusion_matrix(y_test_idx,y_pred_idx,labels=le.classes_,normalize="true")
sns.heatmap(cm,annot=True,fmt='.2f',cmap='Blues',yticklabels=le.classes_,xticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#plot loss e accuracy
plt.figure(figsize=(12,4))
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()