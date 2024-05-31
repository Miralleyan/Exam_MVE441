import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD, NMF, KernelPCA
from sklearn.manifold import TSNE
from matplotlib import colormaps
import matplotlib.pyplot as plt


MyNIST = pd.read_csv("../Exam_MVE441_1/MyMNIST.csv", sep=",")
print
labels = MyNIST["label"]
MyNIST = MyNIST.drop("label", axis=1)

x_train, x_test, y_train, y_test = train_test_split(MyNIST,labels, train_size=0.8, stratify=labels)

print(len(x_train))

#plt.imshow(x_train.iloc[158].to_numpy().reshape(28,28))
#plt.show()

skf = StratifiedKFold(n_splits=10)
for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
    x_fold = x_train.iloc[train_index]
    print(x_fold.shape)
    SVD = TruncatedSVD(n_components=100, algorithm="randomized", n_iter=5, random_state=42)
    x_SVD = SVD.fit_transform(x_fold)
    print(x_SVD.shape)