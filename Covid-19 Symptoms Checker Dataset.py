# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
​
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
​
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
​
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
​
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
​
import seaborn as sns
import pandas as pd
data_main = pd.read_csv('/kaggle/input/covid19-symptoms-checker/Raw-Data.csv')
data_main.shape

data_main.head(10)

data_main.dtypes

pd.isnull(data_main)

data_main.describe(include='all')
country = len(data_main.Country.dropna().unique())
age = len(data_main.Age.dropna().unique())
gender = len(data_main.Gender.dropna().unique())
symptoms = len(data_main.Symptoms.dropna().unique())
esymptoms = len(data_main.Experiencing_Symptoms.dropna().unique())
severity = len(data_main.Severity.dropna().unique())
contact = len(data_main.Contact.dropna().unique())
​
print("Total Combination Possible: ",country * age * gender * symptoms * esymptoms * severity * contact)

import itertools
columns = [data_main.Country.dropna().unique().tolist(),
          data_main.Age.dropna().unique().tolist(),
          data_main.Gender.dropna().unique().tolist(),
          data_main.Symptoms.dropna().unique().tolist(),
          data_main.Experiencing_Symptoms.dropna().unique().tolist(),
          data_main.Severity.dropna().unique().tolist(),
          data_main.Contact.dropna().unique().tolist()]

final_data = pd.DataFrame(list(itertools.product(*columns)), columns=data_main.columns)

final_data.shape

final_data.head(5)

final_data.dtypes

data=pd.read_csv('../input/covid19-symptoms-checker/Cleaned-Data.csv')
data.head(10)

data.groupby(['Severity_Severe'])
data

plt.figure(figsize=(8,6))
data.groupby('Fever').size().plot(color='green',kind='bar')
plt.show()

plt.figure(figsize=(15,10))
data.groupby('Dry-Cough').sum().plot(kind='hist')
plt.show()

import matplotlib.pyplot as plt
df = pd.read_csv('../input/covid19-symptoms-checker/Cleaned-Data.csv')
df.head(100)

len(df)

df.columns

df = df.drop(['Country'], axis = 1)
df = df.drop(['Gender_Transgender'],axis = 1)

df.head()

df = df.dropna()
len(df)
none  = df['Severity_None']
essai = np.array(df)
df.columns

essai[:,18]
essai[:,19]
essai[:,20]
essai[:,21]
symptomes = []
for i in range (316800):
    if essai[i,18] == 1 :
        symptomes.append(0)
    if essai[i,19] == 1 :
        symptomes.append(1)
    if essai[i,20] == 1 :
        symptomes.append(0)
    if essai[i,21] == 1 :
        symptomes.append(2)

len(symptomes)

df = df.drop(['Severity_Mild'],axis = 1)
df = df.drop(['Severity_None'],axis = 1)
df = df.drop(['Severity_Moderate'],axis = 1)
df = df.drop(['Severity_Severe'],axis = 1)
df = df.drop(['None_Sympton'], axis=1)
df = df.drop(['None_Experiencing'], axis=1)
df = df.drop(['Contact_No'],axis = 1)
df = df.drop(['Contact_Yes'], axis =1)
df = df.drop(['Contact_Dont-Know'], axis =1)

df.insert(16, "symptomes", symptomes, True)

len(df.columns)

df.head(100)

from sklearn.model_selection import train_test_split

X = df.drop(['symptomes'], axis=1)
y = df.symptomes
df['symptomes'].value_counts()

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
lr = LogisticRegression()
lr.fit(X_train,y_train)

# Importation des méthodes de mesure de performances
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
y_lr = lr.predict(X_test)
print(confusion_matrix(y_test,y_lr))

print(accuracy_score(y_test,y_lr))

X = df.drop(['symptomes'], axis=1)
y = df.symptomes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
importances = rf.feature_importances_
print(importances)
indices = np.argsort(importances)
print(indices)

rf_score = accuracy_score(y_test, y_rf)
print(rf_score)

cm = confusion_matrix(y_test, y_rf)
print(cm)


plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), df.columns[indices])
plt.title('Importance of the characteristics')
