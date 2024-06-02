import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD, NMF, KernelPCA, SparsePCA
from sklearn.manifold import TSNE
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from tqdm import tqdm



MyNIST = pd.read_csv("../Exam_MVE441_1/MyMNIST.csv", sep=",")

labels = MyNIST["label"]
MyNIST = MyNIST.drop("label", axis=1)

x_train, x_test, y_train, y_test = train_test_split(MyNIST,labels, train_size=0.8, stratify=labels, random_state=42)

componenets = 2

skf = StratifiedKFold(n_splits=10)
for i, (_, batch_index) in enumerate(skf.split(x_train, y_train)):
    x_batch = x_train.iloc[batch_index]
    y_batch = y_train.iloc[batch_index]
    #print(x_batch.shape)

    fig, axs = plt.subplots(2,4)
    #axs[0].imshow(x_batch.iloc[0].to_numpy().reshape(28,28))
    

    ### SVD ###
    SVD = TruncatedSVD(n_components=componenets, algorithm="randomized", n_iter=5, random_state=42)
    x_SVD = SVD.fit_transform(x_batch)

    comp1_SVD = SVD.components_[0,:]#SVD.inverse_transform(np.array([1,0]).reshape(1,-1))
    comp2_SVD = SVD.components_[1,:]#SVD.inverse_transform(np.array([0,1]).reshape(1,-1))

    vmin = min(comp1_SVD.min(), comp2_SVD.min())
    vmax =max(comp1_SVD.max(), comp2_SVD.max())

    im1= axs[0,0].imshow(comp1_SVD.reshape(28,28), cmap=colormaps["Greys"], vmin=vmin, vmax=vmax)
    axs[1,0].imshow(comp2_SVD.reshape(28,28), cmap=colormaps["Greys"], vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=[axs[0,0], axs[1,0]])
    axs[0,0].set_title("Truncated SVD")


    #print(x_SVD[0,:].reshape(1,-1))
    #pic_SVD = SVD.inverse_transform(x_SVD[0,:].reshape(1,-1))
    #axs[1].imshow(pic_SVD.reshape(28,28))


    ### KPCA ###
    KPCA = KernelPCA(n_components=componenets, kernel="linear", random_state=42, fit_inverse_transform=True)
    x_KPCA = KPCA.fit_transform(x_batch)
    
    comp1_KPCA = KPCA.inverse_transform(np.array([1,0]).reshape(1,-1))
    comp2_KPCA = KPCA.inverse_transform(np.array([0,1]).reshape(1,-1))

    vmin = min(comp1_KPCA.min(), comp2_KPCA.min())
    vmax =max(comp1_KPCA.max(), comp2_KPCA.max())

    im2 = axs[0,1].imshow(comp1_KPCA.reshape(28,28), cmap=colormaps["Greys"], vmin =vmin, vmax=vmax)
    axs[1,1].imshow(comp2_KPCA.reshape(28,28), cmap=colormaps["Greys"], vmin =vmin, vmax=vmax)
    fig.colorbar(im2, ax=[axs[0,1], axs[1,1]])
    axs[0,1].set_title("Kernel PCA")

    #pic_KPCA = KPCA.inverse_transform(x_KPCA[0,:].reshape(1,-1))
    #axs[2].imshow(pic_KPCA.reshape(28,28))


    ### NMF ####
    nmf = NMF(n_components=componenets, init="nndsvd", solver="cd")
    x_NMF = nmf.fit_transform(x_batch)
    print(nmf.components_[0])

    comp1_NMF = nmf.components_[0] #nmf.inverse_transform(np.array([1,0]).reshape(1,-1))
    comp2_NMF = nmf.components_[1] #nmf.inverse_transform(np.array([0,1]).reshape(1,-1))

    vmin = min(comp1_NMF.min(), comp2_NMF.min())
    vmax =max(comp1_NMF.max(), comp2_NMF.max())

    im3 =axs[0,2].imshow(comp1_NMF.reshape(28,28), cmap=colormaps["Greys"], vmin=vmin, vmax=vmax)
    axs[1,2].imshow(comp2_NMF.reshape(28,28), cmap=colormaps["Greys"], vmin=vmin, vmax=vmax)

    fig.colorbar(im3, ax=[axs[0,2], axs[1,2]])
    axs[0,2].set_title("NMF")

    ### Sparse PCA ###
    SPCA = SparsePCA(n_components=componenets, alpha=1)
    x_SPCA = SPCA.fit_transform(x_batch)

    print(SPCA.components_)
    print(sum(SPCA.components_[0] !=0))
    print(sum(SPCA.components_ [1]!=0))
    comp1_SPCA = SPCA.components_[0] #nmf.inverse_transform(np.array([1,0]).reshape(1,-1))
    comp2_SPCA = SPCA.components_[1] #nmf.inverse_transform(np.array([0,1]).reshape(1,-1))

    vmin = min(comp1_SPCA.min(), comp2_SPCA.min())
    vmax =max(comp1_SPCA.max(), comp2_SPCA.max())

    im4 = axs[0,3].imshow(comp1_SPCA.reshape(28,28), cmap=colormaps["Greys"], vmin=vmin, vmax=vmax)
    axs[1,3].imshow(comp2_SPCA.reshape(28,28), cmap=colormaps["Greys"], vmin=vmin, vmax=vmax)

    fig.colorbar(im4, ax=[axs[0,3], axs[1,3]])
    axs[0,3].set_title("Sparse PCA")

    #fig.colorbar(cax, ax=[axs[i,j] for i in range(2) for j in range(4)])
    #plt.show()
    #pic_NMF = nmf.inverse_transform(x_NMF[0,:].reshape(1,-1))
    #axs[3].imshow(pic_NMF.reshape(28,28))


    tsne = TSNE(n_components=componenets, random_state=42)
    x_TSNE = tsne.fit_transform(x_batch)




    
    fig, axs = plt.subplots(1,5)

    for j in range(10):
        label = TextPath((0,0), str(j))
        axs[0].scatter(x_SVD[y_batch == j, 0], x_SVD[y_batch == j,1], marker= label, s = 100, label =f"{j}")
        axs[0].set_title("Truncated SVD")
        axs[0].set_xlabel("Component 1")
        axs[0].set_ylabel("Component 2")
    

    for j in range(10):
        label = TextPath((0,0), str(j))
        axs[1].scatter(x_KPCA[y_batch == j, 0], x_KPCA[y_batch == j,1], s=100, marker = label, label =f"{j}")
        axs[1].set_title("Kernel PCA")
        axs[1].set_xlabel("Component 1")
        axs[1].set_ylabel("Component 2")
        #plt.legend()
    #plt.show()

    for j in range(10):
        label = TextPath((0,0), str(j))
        axs[2].scatter(x_NMF[y_batch == j, 0], x_NMF[y_batch == j,1], marker= label, s = 100, label =f"{j}")
        axs[2].set_title("NMF")
        axs[2].set_xlabel("Component 1")
        axs[2].set_ylabel("Component 2")

    for j in range(10):
        label = TextPath((0,0), str(j))
        axs[3].scatter(x_SPCA[y_batch == j, 0], x_SPCA[y_batch == j,1], marker= label, s = 100, label =f"{j}")
        axs[3].set_title("Sparse PCA")
        axs[3].set_xlabel("Component 1")
        axs[3].set_ylabel("Component 2")

    for j in range(10):
        label = TextPath((0,0), str(j))
        axs[4].scatter(x_TSNE[y_batch == j, 0], x_TSNE[y_batch == j,1], marker= label, s = 100, label =f"{j}")
        axs[4].set_title("TSNE")
        axs[4].set_xlabel("Component 1")
        axs[4].set_ylabel("Component 2")

    #plt.legend()
    plt.show()


