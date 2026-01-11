#According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, 
# responsible for approximately 11% of total deaths. 
# This dataset is used to predict whether a patient is likely to get stroke based on the input parameters
# like gender, age, various diseases, and smoking status. Each row in the data provides relavant 
# information about the patient.
#1) id: unique identifier 
# 2) gender: "Male", "Female" or "Other" 
# 3) age: age of the patient
# 4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# 5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease 
# 6) ever_married: "No" or "Yes" 
# 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed" 
# 8) Residence_type: "Rural" or "Urban"
# 9) avg_glucose_level: average glucose level in blood 
# 10) bmi: body mass index 
# 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"* 
# 12) stroke: 1 if the patient had a stroke or 0 if not *Note: "Unknown" in smoking_status means that the information is unavailable for this patient
#classificazione alberi decisionali
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay,classification_report

df=pd.read_csv('healthcare.csv')
print("prime righe: ")
print(df.head())
print("\n\n dimensione: ")
print(df.shape)
print("\n\n descrizioni varie: ")
print(df.describe(include='all'))
print("\n\n tipi di dato: ")
print(df.dtypes)
print("\n\n somma valori nulli per colonna")
print(df.isnull().sum())
print("\n\n informazioni varie")
df.info()

#target
target='stroke'
print('conteggio valori: ')
print(df[target].value_counts())

#grafici esplorativi
#heatmap valori nulli
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("heatmap valori nulli")
plt.show()

#istogrammmi feature
df.hist(figsize=(6,4))
plt.title("istogrammi esplorativi")
plt.show()

#relazioni feature con target (boxplot o violinplot)
feature=['age','hypertension','heart_disease','avg_glucose_level','bmi']
for f in feature:
  plt.figure(figsize=(5,2))
  sns.boxplot(data=df,x=target,y=f)
  plt.title(f'feature {f} vs {target}')
  plt.show()

#matrice di correlazione delle feature numeriche
plt.figure(figsize=(6,4))
df=df.drop('id',axis=1)
corr=df.select_dtypes(include=np.number).corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title("matrice correlazione per eda iniziale")
plt.show()
df2=df.drop(target,axis=1)
sns.heatmap(df2.select_dtypes(include=np.number).corr(),fmt='.2f',annot=True,cmap='coolwarm')
plt.title('matrice correlazione per trovate feature correlate')
plt.show()

#gestione valori nulli e replace di infinti
df.dropna(how='all',inplace=True)
df=df.dropna(subset=target,how='any')
df=df.replace([np.inf,-np.inf],np.nan)
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

#conversione tipi di dati
#eventuali label e one hot encoding, replace o cut
cat_features = ['gender','ever_married','work_type','Residence_type','smoking_status']
df_encoded = pd.get_dummies(
    df,
    columns=cat_features,
    drop_first=True
)

#definizione target e feature
x=df_encoded.drop(target,axis=1)
y=df_encoded[target]

#split dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,stratify=y,test_size=0.3)

#addestramento modello
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

#predict e stampa accuracy
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)
print(f"accuracy : {accuracy:.2f}")
print(f"classification report \n {classification_report(y_pred,y_test)}")

#stampa matrice confusione
labels=sorted(y_train.unique())
cm=confusion_matrix(y_test,y_pred,labels=labels)
disp=ConfusionMatrixDisplay(cm,display_labels=labels)
disp.plot(cmap='Blues')
plt.title("matrice confusione")
plt.show()

#stampa dell'albero con plot_tree
plot_tree(model,max_depth=2,feature_names=x.columns.tolist(),class_names=[str(cls) for cls in labels])
plt.title("albero decisionale")
plt.show()