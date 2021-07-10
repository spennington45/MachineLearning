from data_preprocessing_template import X_test, X_train
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import data
data = pd.read_csv("Data.csv")

# make matrix of data
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print(x)
print(y)

# replace missing data for numerical values only
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:])
x[:, 1:] = imputer.transform(x[:, 1:])

print(x)

# change strings into numbers when there are more then one and on dependant var
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

print(x)

# converts binary strings to numbers yes and not => 0 and 1
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)