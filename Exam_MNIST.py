import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


MyNIST = pd.read_csv("./MyMNIST.csv", sep=",")
print(MyNIST.columns)
labels = MyNIST["label"]

x_train, x_test, y_train, y_test = train_test_split(MyNIST, train_size=0.7, stratify=labels)

print(x_train)