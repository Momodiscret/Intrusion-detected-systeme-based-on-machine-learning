# -*- coding: utf-8 -*-
"""
Created on Sun May 28 02:15:13 2023

@author: Mohamed Ibrahim
"""
# importation des bibliotheques

#data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
#estimateurs
from sklearn.svm import SVC
#multiclass
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
#model selection
from sklearn.model_selection import GridSearchCV
#metrics
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# importation de jeu de donnée d'entrainement et de test
data_test = pd.read_csv("UNSW_NB15_testing.csv")
data_train = pd.read_csv("UNSW_NB15_training.csv")


data_test.info()
data_train['label'].value_counts()
data_train['attack_cat'].value_counts().plot.bar(color = 'red')
"""
# Encodage des valeurs categoriques
categorical= ['proto', 'service', 'state', 'attack_cat']
encoder = LabelEncoder()
for column in categorical:
    data_test[column] = encoder.fit_transform(data_test[column])
    data_train[column] = encoder.fit_transform(data_train[column])
    
# fonction de normalisation des données

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

# Standardisation de X_train et X_test
X_train = MinmaxScal(X_train)
X_test = MinmaxScal(X_test)

# Modele selection Gridsearch
param_grid = {'C':np.logspace(-3,3,7),  
              'gamma':np.logspace(-3,3,7)
              } # elargir l'intervalle (quand on aura une puissance de calcul)
        
clf = SVC()
grid_search = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1, return_train_score=True, verbose=1)
grid_search.fit(X_train, y_train)

best_parame = grid_search.best_params_ # gamma = 0.1, C=100

# entrainement du model avec svm multiclass
clf = SVC(**best_parame)
ovo_clf = OneVsOneClassifier(clf)
ovo_clf.fit(X_train, y_train)
y_pred = ovo_clf.predict(X_test)


#metrics
print(classification_report(y_test, y_pred))

# matrice de confusion
cmt = confusion_matrix(y_test, y_pred)

# plot confusion matrice
cm_df = pd.DataFrame(cmt, index = ['Analysis','Backdoor','Dos','Exploits','Fuzzers','Generics','Normal','Reconnaissance','Shellcode','Worms'], 
                     columns = ['Analysis','Backdoor','Dos','Exploits','Fuzzers','Generics','Normal','Reconnaissance','Shellcode','Worms'])
sns.heatmap(cm_df, annot=True, fmt="g",linewidths=1)
plt.title('Confusion Matrix')
plt.xlabel('Classe prédite')
plt.ylabel('Classe réelle')

















