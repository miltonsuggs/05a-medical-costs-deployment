import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('SUGGS PROJECTS/Medical Costs/insurance.csv')

df['age'] = df['age'].astype('int') #Convert data type of 'age' to int
df['children'] = df['children'].astype('int') #Convert data type of 'children' to int

# sex, female = 0, male = 1
le = LabelEncoder()
le.fit(df.sex.drop_duplicates()) 
df.sex = le.transform(df.sex)

# smoker/non-smoker - non-smoker=0, smoker=1
le.fit(df.smoker.drop_duplicates()) 
df.smoker = le.transform(df.smoker)

#region - Northeast = 0, Northwest = 1, Southeast = 2, Southwest = 3
le.fit(df.region.drop_duplicates()) 
df.region = le.transform(df.region)

X = df.drop(['charges'], axis = 1)
y = df.charges

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

rf = RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = 1, n_jobs = -1)
rf.fit(X_train, y_train)
yhat_rf = rf.predict(X_test)

file = open('model.pkl', 'wb')

pickle.dump(rf, file)

model = pickle.load(open('model.pkl', 'rb'))

! pip3 freeze > requirements.txt