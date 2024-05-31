import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from matplotlib import colormaps
import matplotlib.pyplot as plt


MyNIST = pd.read_csv("../Exam_MVE441_1/MyMNIST.csv", sep=",")
labels = MyNIST["label"]
MyNIST = MyNIST.drop("label", axis=1)

x_train, x_test, y_train, y_test = train_test_split(MyNIST,labels, train_size=0.7, stratify=labels)

#print(x_train)

plt.imshow(x_train.iloc[158].to_numpy().reshape(28,28))
plt.show()