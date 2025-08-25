# Load libraries

import math

from pandas import read_csv

from pandas.plotting import scatter_matrix

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

print("hello world")


# Load dataset

url = "~/Documents/Programming/AIPlayground/MedicalML/DWBCD/breast+cancer+wisconsin+diagnostic/wdbc.data"

# WBC data set has 32 columns, 569 rows
names = ['ID','Diagnosis','radius1','texture1','perimeter1','area1',
         'smoothness1','compactness1','concavity1','concave_points1',
         'symmetry1','fractal_dimension1','radius2','texture2','perimeter2',
         'area2','smoothness2','compactness2','concavity2','concave_points2',
         'symmetry2','fractal_dimension2','radius3','texture3','perimeter3',
         'area3','smoothness3','compactness3','concavity3','concave_points3',
         'symmetry3','fractal_dimension3']

dataset = read_csv(url, names=names)

dataset_chunks = []

chunk_width = 8
num_chunks = math.ceil(len(names)/chunk_width)
for i in range (0,num_chunks):
    print(names[i*chunk_width:(i+1)*chunk_width])
    dataset_chunks.append(dataset[names[i*chunk_width:(i+1)*chunk_width]])
    

# print(dataset)


# shape
print("***** Shape:")
for d in dataset_chunks:
    print(d.shape)

# head
print("***** Head:")
for d in dataset_chunks:
    print(d.head(20))

# descriptions
print("***** Descriptions:")
for d in dataset_chunks:
    print(d.describe())

# class distribution
print("***** Class Distribution:")
print(dataset.groupby('Diagnosis').size())
print("best guess is the dtype: int64 means the size values (i.e. the number 357) are of type int64")

# playing with what is the type output after it
print(dataset.groupby('Diagnosis'))
print(dataset.groupby('Diagnosis').mean())
print(dataset.groupby('Diagnosis')["radius1"].mean())