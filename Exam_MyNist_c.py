import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import k_means, SpectralClustering, HDBSCAN, AgglomerativeClustering
from matplotlib.text import TextPath
import matplotlib.pylab as plt

MyNIST = pd.read_csv("../Exam_MVE441_1/MyMNIST.csv", sep=",")

labels = MyNIST["label"]
MyNIST = MyNIST.drop("label", axis=1)

x_train, x_test, y_train, y_test = train_test_split(MyNIST,labels, train_size=0.8, stratify=labels, random_state=42)

#x_TSNE_df = TSNE(2).fit_transform(x_train)
x_TSNE_df = pd.read_csv("./TSNE_data_2", index_col=0)

max_cluster = 25

clusters = pd.DataFrame()
sihouette = pd.DataFrame()
sil = np.zeros((2,len(x_train), max_cluster-1))

skf = StratifiedKFold(n_splits=10)
for i, (_, batch_index) in enumerate(skf.split(x_train, y_train)):
    print(f"Batch: {i}")
    x_batch = x_train.iloc[batch_index]
    y_batch = y_train.iloc[batch_index]
    x_TSNE = x_TSNE_df.iloc[batch_index].to_numpy()


    inert = np.zeros((max_cluster-1))
    silhouette_width = np.zeros((max_cluster-1, 2))

    for n_cluster in range(2,max_cluster + 1):
        print("cluster: ", n_cluster)
        
        AC = AgglomerativeClustering(n_clusters=n_cluster).fit(x_batch)
        label_AC = AC.labels_
        centroid, label_kmean, inertia = k_means(X=x_batch, n_clusters=n_cluster, n_init="auto")
        sil[0, 4000*i:4000*(i+1), n_cluster-2] = silhouette_samples(x_batch, label_kmean)
        sil[1, 4000*i:4000*(i+1), n_cluster-2] = silhouette_samples(x_batch, label_AC)

        #inert[n_cluster-2] = inertia
        #print(silhouette_samples(x_batch, label_kmean))
        #silhouette_width[n_cluster-2,0] = silhouette_score(x_batch, label_kmean)
        #silhouette_width[n_cluster-2,1] = silhouette_score(x_batch, label_AC)
        
        #silhouette_width[n_cluster-2,1] = silhouette_score(x_batch, label_SC)

    #plt.scatter(list(range(2,max_cluster+1)), inert)
    #plt.ylabel("Inertia")
    #plt.xlabel("Number of clusters")
    """
    model = ["Agglomerative clustering", "KMeans"]
    for j in range(2):
        plt.plot(list(range(2,max_cluster+1)), silhouette_width[:,j], label = model[j])
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhoutte width")
    plt.legend()
    plt.show()
    """

mean_sil_KNN = np.zeros((max_cluster-1))
mean_sil_AC = np.zeros((max_cluster-1))

pd.DataFrame(data = sil[0,:,:]).to_csv("./silhouette_KNN")
pd.DataFrame(data = sil[1,:,:]).to_csv("./silhouette_AC")

for j in range(max_cluster-1):
    mean_sil_KNN[j] = sil[0,:, j].mean()
    mean_sil_AC[j] = sil[1,:, j].mean()


plt.plot(list(range(2,max_cluster+1)), mean_sil_KNN, label = "KNN")
plt.plot(list(range(2,max_cluster+1)), mean_sil_AC, label = "Agglomerative clustering")
plt.xlabel("Number of clusters")
plt.ylabel("Mean Silhoutte width")
plt.legend()
plt.show()


quit()
skf = StratifiedKFold(n_splits=10)
for i, (_, batch_index) in enumerate(skf.split(x_train, y_train)):
    x_batch = x_train.iloc[batch_index]
    y_batch = y_train.iloc[batch_index]
    x_TSNE = x_TSNE_df.iloc[batch_index].to_numpy()

    label_HD = HDBSCAN(min_cluster_size=10).fit_predict(x_batch)
    #print(labels)
    #print(max(labels))
    n_cluster = 15
    label_AC = AgglomerativeClustering(n_clusters=n_cluster).fit(x_batch).labels_
    centroid, label_kmean, inertia = k_means(X=x_batch, n_clusters=n_cluster, n_init="auto")

    fig, axs = plt.subplots(1,3)
    print(max(label_kmean))
    for j in range(max(label_kmean)+1):
        axs[0].scatter(x_TSNE[label_kmean == j, 0], x_TSNE[label_kmean == j,1], label = f"Label {j}")
        axs[0].set_title("Kmean")
        axs[0].set_xlabel("Component 1")
        axs[0].set_ylabel("Component 2")

    print(max(label_AC))
    for j in range(max(label_AC)+1):
        axs[1].scatter(x_TSNE[label_AC == j, 0], x_TSNE[label_AC == j,1], label = f"Label {j}")
        axs[1].set_title("Agglomerative clustering")
        axs[1].set_xlabel("Component 1")
        axs[1].set_ylabel("Component 2")

    print(max(label_HD))
    for j in range(max(label_HD)+1):
        axs[2].scatter(x_TSNE[label_HD == j, 0], x_TSNE[label_HD == j,1], label = f"Label {j}")
        axs[2].set_title("HDBSCAN")
        axs[2].set_xlabel("Component 1")
        axs[2].set_ylabel("Component 2")
    #plt.legend()
    plt.show()

