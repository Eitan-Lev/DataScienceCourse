import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import set_option
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(filename, names=names)
set_option('display.width', 100)
set_option('precision', 3)
description = df.describe()
print(description)

# Some Info
df.isna().sum()
df.isna().sum()/len(df)*100
print(df.info())
print (df.describe())
print(df.corr())

# zscore
from scipy import stats
df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]


# Distribution
from pandas import read_csv
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(filename, names=names)
class_counts = df.groupby('class').size()
print(class_counts)

# Correlations
from pandas import read_csv
from pandas import set_option
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(filename, names=names)
set_option('display.width', 100)
set_option('precision', 3)
correlations = df.corr(method='pearson')
print(correlations)

#Skew
from pandas import read_csv
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = read_csv(filename, names=names)
skew = dataset.skew()
print(skew)

# Importing the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
print(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:8])
X[:, 1:8] = imputer.transform(X[:, 1:8])
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print (X_train)
print (X_test)
print (X_train)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# pandas_profiling

from pandas_profiling import ProfileReport
profile = ProfileReport(dataset, title="Pandas Profiling Report")

profile.to_widgets()

profile.to_notebook_iframe()

profile.to_file("your_report.html")