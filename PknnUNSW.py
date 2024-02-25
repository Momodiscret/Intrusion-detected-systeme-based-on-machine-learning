# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:12:04 2023

@author: Mohamed Ibrahim
"""
# importation des bibliotheques

#data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
#estimateurs
from sklearn.neighbors import KNeighborsClassifier
#multiclass
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
#model selection
from sklearn.model_selection import GridSearchCV
#metrics
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# importation de jeu de donnée d'entrainement et de test
data_test = pd.read_csv("UNSW_NB15_testing.csv")
data_train = pd.read_csv("UNSW_NB15_training.csv")


data_train.info()
data_train['label'].value_counts().plot.bar()
data_train['attack_cat'].value_counts().plot.bar(color = 'red')
data_test['attack_cat'].value_counts().plot.bar(color = 'red')

# Encodage des valeurs categoriques
categorical= ['proto', 'service', 'state', 'attack_cat']
encoder = LabelEncoder()
for column in categorical:
    data_test[column] = encoder.fit_transform(data_test[column])
    data_train[column] = encoder.fit_transform(data_train[column])

    
# fonction pour redimensionner les données

def MinmaxScal(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

      
#sns.pairplot(data_train, hue="attack_cat")
# sns.heatmap(data_train)

# Données d'entrainement et de tests
X_train = data_train.drop("id", axis='columns').iloc[:, :-2].values
y_train = data_train['attack_cat'].values
X_test = data_test.drop("id", axis='columns').iloc[:, :-2].values
y_test = data_test['attack_cat'].values

# normalisation de X_train et X_test

X_train =  MinmaxScal(X_train)
X_test =  MinmaxScal(X_test)

#-------------KNN----------------------
# Modele selection Gridsearch
parameters = {'n_neighbors' : list(range(5, 51))} 
classifier = KNeighborsClassifier() 
grid_sear = GridSearchCV(classifier, parameters, cv=10, verbose=0) 
grid_sear.fit(X_train, y_train)

best_param = grid_sear.best_params_
best_param # 23

# entrainement du model avec KNN
knn_clf = KNeighborsClassifier(**best_param)
Ov_clf =  OneVsRestClassifier(knn_clf)
Ov_clf.fit(X_train, y_train)

y_pred = Ov_clf.predict(X_test)

# metrics

print(classification_report(y_test, y_pred))

# matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# plot confusion matrice
cmt_df = pd.DataFrame(cm, index = ['Analysis','Backdoor','Dos','Exploits','Fuzzers','Generics','Normal','Reconnaissance','Shellcode','Worms'], 
                     columns = ['Analysis','Backdoor','Dos','Exploits','Fuzzers','Generics','Normal','Reconnaissance','Shellcode','Worms'])
sns.heatmap(cmt_df, annot=True, fmt="g",linewidths=1)
plt.title('Confusion Matrix')
plt.xlabel('Classe prédite')
plt.ylabel('Classe réelle')

#-------------------------------------------------------


