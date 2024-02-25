# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:11:40 2023

@author: Mohamed Ibrahim
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


df = pd.read_csv("Train_data.csv")
df.info()

df['class'].value_counts().plot.bar()

categorical= ['protocol_type', 'service', 'flag', 'class']
encoder = LabelEncoder()
for column in categorical:
    df[column] = encoder.fit_transform(df[column])

sns.pairplot(df, hue="class", markers=["o", "s"])

X = df[df.columns[:-1]].values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

classifier = SVC(kernel = 'rbf', gamma='scale', random_state= 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cmt = confusion_matrix(y_test, y_pred)
cmt

display = ConfusionMatrixDisplay(confusion_matrix = cmt, display_labels = [False, True])
display.plot()

print(classification_report(y_test, y_pred))


