
import math

from pandas import read_csv

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

DEBUG = False

# Load dataset

def load_dataset():
    url = "~/Documents/Programming/AIPlayground/MedicalML/DWBCD/breast+cancer+wisconsin+diagnostic/wdbc.data"

    # WBC data set has 32 columns, 569 rows
    names = ['ID','Diagnosis','radius1','texture1','perimeter1','area1',
            'smoothness1','compactness1','concavity1','concave_points1',
            'symmetry1','fractal_dimension1','radius2','texture2','perimeter2',
            'area2','smoothness2','compactness2','concavity2','concave_points2',
            'symmetry2','fractal_dimension2','radius3','texture3','perimeter3',
            'area3','smoothness3','compactness3','concavity3','concave_points3',
            'symmetry3','fractal_dimension3']
    feature_names = [n for n in names if n not in ['ID','Diagnosis']]
    dataset = read_csv(url, names=names)

    dataset_features_only = dataset.drop(columns=['ID','Diagnosis'])

    dataset_chunks = []
    dataset_chunks_features_only = []

    chunk_width = 4
    num_chunks = math.ceil(len(names)/chunk_width)
    for i in range (0,num_chunks):
        print(names[i*chunk_width:(i+1)*chunk_width])
        dataset_chunks.append(dataset[names[i*chunk_width:(i+1)*chunk_width]])
        dataset_chunks_features_only.append(dataset_features_only[feature_names[i*chunk_width:(i+1)*chunk_width]])
        
    
    return (dataset, dataset_features_only, dataset_chunks, dataset_chunks_features_only)


(dataset, dataset_features_only, dataset_chunks, dataset_chunks_features_only) = load_dataset()

# issue 7.2.2 normalization:
dataset_scaled = dataset.copy() 
  
# apply normalization techniques 
for column in dataset_scaled.columns: 
    try:
        dataset_scaled[column] = (dataset_scaled[column] -
                           dataset_scaled[column].mean()) / dataset_scaled[column].std()
    except Exception as e:
        print("EXCEPTION:")
        print(e)
        print("END OF EXCEPTION")
        # don't scale if there is an exception, as happens with Diagnosis column
  
# view normalized data to check if works, when doing Z score normalization, all seem to max out in 100-200 range, better than some being in 1000's and some being in 0.01's   
# dataset_scaled.hist()

# Split-out validation dataset

if DEBUG:
    print(dataset_scaled)

array = dataset_scaled.values

if DEBUG:
    print(array) # correct

X = array[:,2:32] # X is columns 3 to 32

if DEBUG:
    print(X) # correct

y = array[:,1] # y is 2nd column, index 1

if DEBUG:
    print(y) # correct

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

if DEBUG:
    print(X_train) 
    print(Y_train) # first 3 train examples correspond between X and Y
    print(X_validation)
    print(Y_validation) # first 3 validation examples correspond between X and Y
    print(len(X),len(X_train),len(Y_train),len(X_validation),len(Y_validation)) # lengths are 20% and 80% as they should be








#issue 5.3 section:

# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # test: everything but CART does the same when I run it twice


# issue 5.4

# Compare Algorithms

SHOW7_5_2_1 = False
if SHOW7_5_2_1:
    plt.boxplot(results, labels=names)

    plt.title('Algorithm Comparison')

    plt.show()



# section 7.2.5.2.1 (6b.1)

# Make predictions on validation dataset

model = LogisticRegression(solver='liblinear', multi_class='ovr')

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)

print(X_validation)
print(predictions) # seems correctish now that I am using the correct data (mix of M and B). On first 3 examples, they were all correctly classified

# issue 7.2.5.2.2 (6b.2)

# Evaluate predictions

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))