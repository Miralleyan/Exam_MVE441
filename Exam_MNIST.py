import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD, NMF, KernelPCA
from sklearn.manifold import TSNE
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from tqdm import tqdm


MyNIST = pd.read_csv("../Exam_MVE441_1/MyMNIST.csv", sep=",")
print
labels = MyNIST["label"]
MyNIST = MyNIST.drop("label", axis=1)

x_train, x_test, y_train, y_test = train_test_split(MyNIST,labels, train_size=0.8, stratify=labels)

componenets = 10

skf = StratifiedKFold(n_splits=5)
for i, (_, batch_index, ) in enumerate(skf.split(x_train, y_train)):
    x_batch = x_train.iloc[batch_index]
    y_batch = y_train.iloc[batch_index]
    #print(x_batch.shape)

    fig, axs = plt.subplots(1,4)
    axs[0].imshow(x_batch.iloc[0].to_numpy().reshape(28,28))
  

    SVD = TruncatedSVD(n_components=componenets, algorithm="randomized", n_iter=5, random_state=42)
    x_SVD = SVD.fit_transform(x_batch)

    #print(x_SVD[0,:].reshape(1,-1))
    pic_SVD = SVD.inverse_transform(x_SVD[0,:].reshape(1,-1))
    axs[1].imshow(pic_SVD.reshape(28,28))
 


    KPCA = KernelPCA(n_components=componenets, kernel="linear", random_state=42, fit_inverse_transform=True)
    x_KPCA = KPCA.fit_transform(x_batch)

    pic_KPCA = KPCA.inverse_transform(x_KPCA[0,:].reshape(1,-1))
    axs[2].imshow(pic_KPCA.reshape(28,28))


    nmf = NMF(n_components=componenets, init="nndsvda", solver="mu")
    x_NMF = nmf.fit_transform(x_batch)

    pic_NMF = nmf.inverse_transform(x_NMF[0,:].reshape(1,-1))
    axs[3].imshow(pic_NMF.reshape(28,28))
    plt.show()

    """
    fig, axs = plt.subplots(1,3)

    for j in range(10):
        label = TextPath((0,0), str(j))
        axs[0].scatter(x_KPCA[y_batch == j, 0], x_KPCA[y_batch == j,1], s=100, marker = label, label =f"{j}")
        #plt.legend()
    #plt.show()

    for j in range(10):
        label = TextPath((0,0), str(j))
        axs[1].scatter(x_SVD[y_batch == j, 0], x_SVD[y_batch == j,1], marker= label, s = 100, label =f"{j}")
    
    for j in range(10):
        label = TextPath((0,0), str(j))
        axs[2].scatter(x_NMF[y_batch == j, 0], x_NMF[y_batch == j,1], marker= label, s = 100, label =f"{j}")

    plt.legend()
    plt.show()
    quit()
"""
'''
error = []
n1 = 1
n2 = 783
SVD = TruncatedSVD(n_components=x_train.shape[1]-1, algorithm="randomized", n_iter=5, random_state=42)
x_SVD = SVD.fit_transform(x_train)
for q in tqdm(range(n1,n2+1)):
    #print([np.linalg.norm(x_SVD[:,i])**2 for i in range(q+1,x_train.shape[1]-1)])
    error.append(sum([np.linalg.norm(x_SVD[:,i])**2 for i in range(q+1,x_SVD.shape[1]-1)]))
    #for i in range(x_SVD.shape[1]):
    #    print(np.linalg.norm(x_SVD[:,i]))
    #print(SVD.singular_values_)
    #print(x_SVD)
    #print(x_SVD.shape)

pd.DataFrame
print(error)
plt.plot([i for i in range(n1,n2+1)], error)
plt.show()'''