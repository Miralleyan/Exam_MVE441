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

colors = ["Blue", "orange", "Green", "Red", "Purple", "Brown", "pink", "gray", "Olive", "cyan"]

MyNIST = pd.read_csv("../Exam_MVE441_1/MyMNIST.csv", sep=",")

labels = MyNIST["label"]
MyNIST = MyNIST.drop("label", axis=1)

x_train, x_test, y_train, y_test = train_test_split(MyNIST,labels, train_size=0.8, stratify=labels, random_state=42)

components = 2

if exists(f"TSNE_data_{components}"):
    x_TSNE = pd.read_csv(f"TSNE_data_{components}", index_col= 0)
    print("Imported TSNE data")
else:
    tsne = TSNE(n_components=components, random_state=42)
    x_TSNE = pd.DataFrame(data = tsne.fit_transform(x_train), columns= x_train.columns[:components], index= x_train.index)
    x_TSNE.to_csv(f"./TSNE_data_{components}", sep = ",")
    print("Calculated TSNE")



for size in [0.01, 0.05, 0.1, 0.25, 0.4, 0.55, 0.7]:
    y_pred = pd.DataFrame()

    skf = StratifiedKFold(n_splits=10)
    for i, (train_index, val_index) in enumerate(skf.split(x_TSNE, y_train)):
        x_ = x_TSNE.iloc[train_index]
        y_ = y_train.iloc[train_index]

        x_val = x_TSNE.iloc[val_index]
        y_val = y_train.iloc[val_index]
        val_size = len(val_index)


        pred_KNN = {ind:[0]*10 for ind in x_val.index}
        pred_RF = {ind:[0]*10 for ind in x_val.index}
        pred_SVC = {ind:[0]*10 for ind in x_val.index}
        pred_LR = {ind:[0]*10 for ind in x_val.index}
        pred_LDA = {ind:[0]*10 for ind in x_val.index}

        batches = StratifiedKFold(n_splits=10)
        for j, (_, batch_index) in enumerate(batches.split(x_, y_)):
            print(f"Size {size}, fold {i+1}/10, batch {j+1}/10")
            x_batch = x_.iloc[batch_index]
            y_batch = y_.iloc[batch_index]

            x_size, _ , y_size, _ = train_test_split(x_batch, y_batch, train_size= size/0.72, stratify=y_batch, random_state=42)

            #### KNN ####
            KNN = KNeighborsClassifier(n_neighbors = 5)
            KNN.fit(x_size,y_size)
            y_predict_KNN = KNN.predict(x_val)
            prob_KNN = KNN.predict_proba(x_val)
     
            

            #### RF ####
            RF = RandomForestClassifier(n_estimators=500)
            RF.fit(x_size,y_size)
            y_predict_RF = RF.predict(x_val)
            prob_RF = RF.predict_proba(x_val)



            #### SVC ####
            svc = SVC(kernel="rbf")
            svc.fit(x_size, y_size)
            y_predict_SVC = svc.predict(x_val)



            #### LR ####
            LR = LogisticRegression(penalty="elasticnet", class_weight="balanced", solver="saga", l1_ratio=0.5, max_iter=200)
            LR.fit(x_size, y_size)
            y_predict_LR = LR.predict(x_val)



            #### LDA ####
            LDA = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
            LDA.fit(x_size, y_size)
            y_predict_LDA = LDA.predict(x_val)


            for ind in range(len(x_val)):
                pred_KNN[x_val.iloc[ind].name][y_predict_KNN[ind]] += 1
                pred_RF[x_val.iloc[ind].name][y_predict_RF[ind]] += 1
                pred_SVC[x_val.iloc[ind].name][y_predict_SVC[ind]] += 1
                pred_LR[x_val.iloc[ind].name][y_predict_LR[ind]] += 1
                pred_LDA[x_val.iloc[ind].name][y_predict_LDA[ind]] += 1
        
        y_pred_KNN = pd.DataFrame(data = [np.argmax(pred_KNN[ind]) for ind in x_val.index], columns=["KNN"], index=[ind for ind in x_val.index])
        y_pred_RF = pd.DataFrame(data = [np.argmax(pred_RF[ind]) for ind in x_val.index], columns=["RF"], index=[ind for ind in x_val.index])
        y_pred_SVC = pd.DataFrame(data = [np.argmax(pred_SVC[ind]) for ind in x_val.index], columns=["SVC"], index=[ind for ind in x_val.index])
        y_pred_LR = pd.DataFrame(data = [np.argmax(pred_LR[ind]) for ind in x_val.index], columns=["LR"], index=[ind for ind in x_val.index])
        y_pred_LDA = pd.DataFrame(data = [np.argmax(pred_LDA[ind]) for ind in x_val.index], columns=["LDA"], index=[ind for ind in x_val.index])


        y_pred = y_pred._append(pd.concat([y_val, y_pred_KNN, y_pred_RF, y_pred_SVC, y_pred_LR, y_pred_LDA], axis = 1))

    y_pred.to_csv(f"./Data/2_y_pred_{size}")
#y_pred_mat = y_pred_mat._append(pd.concat([KNN_y_pred, QDA_y_pred, LR_y_pred, RF_y_pred, SVC_y_pred, LDA_y_pred], axis=1))
#class_prob = class_prob._append((KNN_prob + QDA_prob + LR_prob + RF_prob + SVC_prob + LDA_prob)/6)

#pred_batch = pred_batch._append(pd.DataFrame(data = np.array([y_predict_KNN, y_predict_RF]).T, columns= ["KNN", "RF"], index=[l for l in range(val_size*j,val_size*(j+1))]))
#print(pred_batch)

"""
for j in range(10):
    label = TextPath((0,0), str(j))
    plt.scatter(x_TSNE.to_numpy()[y_train.to_numpy() == j, 0], x_TSNE.to_numpy()[y_train.to_numpy() == j,1], marker= label, c = colors[j], s = 100, label =f"{j}")
    plt.scatter(x_val.to_numpy()[y_pred_KNN["KNN"] == j, 0], x_val.to_numpy()[y_pred_KNN["KNN"] == j,1], marker= label, c = "black", s = 200, label =f"{j}")
plt.show()
"""
        
            



