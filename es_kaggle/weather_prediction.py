#Explore the intricate climate patterns of the "Emerald City" with the Seattle Weather Dataset. 
# Dive into a comprehensive collection of weather data that unveils the city's renowned reputation for rain and its ever-changing atmospheric conditions. 
# Uncover seasonal trends, precipitation variations, and temperature fluctuations, all encapsulating the unique charm of Seattle's weather. 
# Whether you're a data enthusiast, a climate researcher, or simply curious about the city's meteorological nuances, 
# this dataset provides valuable insights into Seattle's dynamic weather landscape.


#prevedere la variabile qualitativa weather con alberi decisionali
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay

df=pd.read_csv("weather.csv")
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.info())
print(df.describe(include='all'))

target='weather'
print(df[target].value_counts())

#grafici esplorativi
#istogrammi feature
df.hist(figsize=(4,6))
plt.suptitle("istogrammi esplorativi")
plt.show()

#mappa valori null
print(df.isnull().sum())
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("heatmap valori nulli")
plt.show()

#heatmap correlazione feature numeriche o target e feature
df_num=df.select_dtypes(include=np.number)
sns.heatmap(df_num.corr(),cmap='coolwarm',annot=True)
plt.title("heatmap correlazione feature numeriche")
plt.show()

#conversione tipi di dato e eliminazione colonne inutili
df['date']=pd.to_datetime(df['date'],errors='coerce')
df['month']=df['date'].dt.month
df.drop('date',axis=1)

#boxplot feature vs target
feature=['precipitation','temp_max','temp_min','wind','month']
for col in feature:
  sns.boxplot(x=target,y=col,data=df)
  plt.title(f"boxplot {col} vs {target}")
  plt.show()

#gestione valori nulli
df=df.dropna(axis=1,how='all')
df.dropna(subset=target,inplace=True)



#encoding vari
le=LabelEncoder()
df['month']=le.fit_transform(df['month'].astype(str))

#gestione valori nulli feature e raggruppamento feature
df=df.replace([np.inf,-np.inf],np.nan)
feature=['precipitation','temp_max','temp_min','wind','month']
df=df.dropna(subset=feature,how='any')

#split
x=df[feature]
y=df[target]
x_test,x_train,y_test,y_train=train_test_split(x,y,stratify=y,random_state=42,test_size=0.3)

#scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#addestramento
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

#predict e metriche
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)
print(f"accuracy: {accuracy:.2f}")
print(classification_report(y_test,y_pred))

#matrice di confusione
labels=sorted(y_train.unique())
cm=confusion_matrix(y_test,y_pred,labels=labels)
disp=ConfusionMatrixDisplay(cm,display_labels=labels)
disp.plot(cmap='Blues')
plt.title("matrice di confusione")
plt.show()

#albero plottato
plot_tree(model,max_depth=2,feature_names=x.columns.tolist(),class_names=[str(cls) for cls in labels])
plt.title("albero decisionale")
plt.show()