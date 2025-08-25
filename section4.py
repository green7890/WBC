

# visualize the data

import math

from pandas import read_csv

from pandas.plotting import scatter_matrix

from matplotlib import pyplot as plt

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


# box and whisker plots

# I have 30 features, layout=(rows,columns)

showBoxAndWhisker = False
if showBoxAndWhisker:
    first_iter = True
    for chunk in dataset_chunks_features_only:
        if first_iter:
            first_iter = False
        else:
            # plt.figure()
            pass
        chunk.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        

    plt.show()

# histograms
showHistogram = False
if showHistogram:
    dataset.hist()

    plt.show()



# scatter plot matrix

scatter_matrix(dataset)

plt.show()