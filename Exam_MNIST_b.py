import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from os.path import exists


MyNIST = pd.read_csv("../Exam_MVE441_1/MyMNIST.csv", sep=",")

labels = MyNIST["label"]
MyNIST = MyNIST.drop("label", axis=1)

x_train, x_test, y_train, y_test = train_test_split(MyNIST,labels, train_size=0.8, stratify=labels, random_state=42)
components = 3

if exists(f"TSNE_data_{components}"):
    x_TSNE = pd.read_csv(f"TSNE_data_{components}", index_col= 0)
    print("Imported TSNE data")
else:
    tsne = TSNE(n_components=components, random_state=42)
    x_TSNE = pd.DataFrame(data = tsne.fit_transform(x_train), columns= x_train.columns[:components], index= x_train.index)
    x_TSNE.to_csv(f"./TSNE_data_{components}", sep = ",")
    print("Calculated TSNE")

batches = StratifiedKFold(n_splits=10)
for i, (_, batch_index) in enumerate(batches.split(x_TSNE, y_train)):
    x_batch = x_TSNE.iloc[batch_index]
    y_batch = y_train.iloc[batch_index]

    skf = StratifiedKFold(n_splits=10)
    for j, (train_index, test_index) in enumerate(skf.split(x_batch, y_batch)):
        print(f"Batch {i} fold {j}")
        x = x_batch.iloc[train_index]
        y = y_batch.iloc[train_index]
        x_val = x_batch.iloc[test_index]
        y_val = y_test.iloc[test_index]

        KNN = KNeighborsClassifier(n_neighbors = 5)
        KNN.fit(x,y)

        y_predict = KNN.predict(x_val)
        #print(y_predict)
        #print(y_val)
        
        print(sum(y_predict == y_val)/len(y_val))


