from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, RFE, SequentialFeatureSelector, chi2, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats


## Load data
fish_df = pd.read_csv("./Fish3.txt", sep=" ")
colors = ["Blue", "Red", "Yellow", "Green", "Purple", "Black", "Pink"]


## Modify it to easier form
labenc = LabelEncoder()
labenc.fit(fish_df["Species"])


fishes = labenc.classes_
fish_label = labenc.transform(fish_df["Species"])
fish_df = fish_df.drop(labels="Species" , axis=1)

for i in range(7):
    print(f"There are {sum(fish_label==i)} {fishes[i]}")


## Split test data for later
x_train, x_test, y_train, y_test = train_test_split(fish_df, fish_label, train_size=0.7, random_state=42)
x_train_np = x_train.to_numpy()


#Scaling data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train_np)

transformations = ["SK_scaled", "SK", "PCA_scaled", "PCA"]
index = [i for i in range(1,37)]
index.insert(0, "KNN"), index.insert(7, "QDA"), index.insert(14, "LR"),index.insert(21, "RF"), index.insert(28, "SVC"), index.insert(35, "LDA")
end_result = pd.DataFrame(data = np.zeros((42,4)),index=index, columns=transformations)

#Cross validation
skf = StratifiedKFold(n_splits=10, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(x_train_np, y_train)):
    print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")


    result = pd.DataFrame(index=index, columns=transformations)
    for k in range(2,7):
        SK_scaled = SelectKBest(chi2, k=k)
        x_sk_scaled = SK_scaled.fit_transform(x_scaled+abs(np.min(x_scaled)), y_train)
        #print(SK.get_feature_names_out())

        SK = SelectKBest(chi2, k=k)
        x_sk = SK.fit_transform(x_train_np+abs(np.min(x_train_np)), y_train)

        pca_scaled = PCA(n_components=k)
        x_pca_scaled = pca_scaled.fit_transform(x_scaled)

        pca = PCA(n_components=k)
        x_pca = pca.fit_transform(x_train_np)

        x = [x_sk_scaled[train_index], x_sk[train_index], x_pca_scaled[train_index], x_pca[train_index]]
        x_val = [x_sk_scaled[test_index], x_sk[test_index], x_pca_scaled[test_index], x_pca[test_index]]
        y = y_train[train_index]
        y_val = y_train[test_index]

        ## KNeighbours, non-linear, non-parametric
        KNN = KNeighborsClassifier(n_neighbors=5)

        ## QDA, non-linear, parametric
        QDA = QuadraticDiscriminantAnalysis()

        ## LR, linear, parametric
        LR = LogisticRegression(penalty="elasticnet", class_weight="balanced", solver="saga", l1_ratio=0.5)

        ## RF, non-linear, non-parametric, 
        RF = RandomForestClassifier(n_estimators=100)

        ## SVC, non-linear, non-parametric
        svc = SVC(kernel="rbf", class_weight="balanced")

        ## LDA, linear, parametric
        LDA = LinearDiscriminantAnalysis()

        ## SVC, linear, non-parametric
        #svc_lin = SVC(kernel="linear", class_weight="balanced")
        
        for j in range(len(x)):
            KNN.fit(x[j], y)
            result.at[k,transformations[j]] = f1_score(KNN.predict(x_val[j]), y_val, average='weighted')

            QDA.fit(x[j], y)
            result.at[k+6,transformations[j]] = f1_score(QDA.predict(x_val[j]), y_val, average='weighted')

            LR.fit(x[j], y)
            #result.at[k,transformations[j]] = accuracy_score(y_pred, y_val)
            result.at[k+12,transformations[j]] = f1_score(LR.predict(x_val[j]), y_val, average='weighted')

            RF.fit(x[j], y)
            result.at[k+18,transformations[j]] = f1_score(RF.predict(x_val[j]), y_val, average='weighted')

            svc.fit(x[j], y)
            result.at[k+24,transformations[j]] = f1_score(svc.predict(x_val[j]), y_val, average='weighted')

            LDA.fit(x[j], y)
            result.at[k+30,transformations[j]] = f1_score(LDA.predict(x_val[j]), y_val, average='weighted')

            #svc_lin.fit(x[j], y)
            #result.at[k+30,transformations[j]] = f1_score(svc_lin.predict(x_val[j]), y_val, average='weighted')

    end_result += result
    print(result)
print(end_result/10)