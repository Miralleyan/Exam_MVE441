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
from matplotlib.text import TextPath
import matplotlib.pylab as plt

MyNIST = pd.read_csv("../Exam_MVE441_1/MyMNIST.csv", sep=",")

labels = MyNIST["label"]
MyNIST = MyNIST.drop("label", axis=1)

x_train, x_test, y_train, y_test = train_test_split(MyNIST,labels, train_size=0.8, stratify=labels, random_state=42)


