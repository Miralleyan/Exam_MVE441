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
import time

## Functions
def plott(x, k):
    fig, axs = plt.subplots(k,k)
    for j in range(k):
        for l in range(k):
            for color, i, species in zip(colors, range(7), fishes):
                axs[j,l].scatter(x[y_train == i, j],x[y_train == i, l], color = color, label = species)
                #plt.title(f"PCA {j} and {k}")
    plt.legend()
    plt.show()

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

# Choose which part of the code to run

corr = False

######################################################################################################################
##this code classfied using 6 different classifiers


### Adding extra features ###
add = 200
for run in range(1,16):
    print(f"Now have {add*run} features")

    ## Not correlated ##
    new_features = pd.concat([pd.DataFrame(data = stats.norm(loc =stats.norm(scale = 4).rvs(), scale = 3).rvs(size = len(x_train)), index=x_train.index , columns=[f"S_{(run-1)*add+i}"]) for i in range(add)], axis=1)

    ## Correlated ##
    new_features_corr = pd.concat([pd.DataFrame(data = x_train[x_train.columns[i%6]].to_numpy()*stats.norm(scale = 2).rvs() +stats.norm(scale = 10).rvs(size = len(x_train)), index=x_train.index , columns=[f"S_{(run-1)*add+i}"]) for i in range(add)], axis=1)
    if corr == 1:
        x_train = pd.concat([x_train,new_features_corr], axis = 1)
    else:
        x_train = pd.concat([x_train, new_features], axis = 1)

    #if run == 1:
    #    continue

    ### Scaling data ###
    scaler = StandardScaler()
    x_scaled_np = scaler.fit_transform(x_train)
    x_scaled = pd.DataFrame(data = x_scaled_np, index=x_train.index, columns= x_train.columns)


    for n in range(1,7):
        print(f"Features {n}:")
        feature_scores = pd.DataFrame()

        skf = StratifiedKFold(n_splits=10)
        for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
            print(f"Outer fold {i}:")
            x = x_scaled.iloc[train_index]
            x_val = x_scaled.iloc[test_index]
            y = y_train[train_index]
            y_val = y_train[test_index]
                    
            
            ### KNN ###
            KNN = KNeighborsClassifier(n_neighbors=5)

            ## Features selection ##
            SK_KNN = SelectKBest(f_classif, k=n)
            SK_KNN.fit(x,y)
            #x_KNN_val = SK_KNN.transform(x_val)
            #KNN.fit(x_KNN, y)


            KNN_features = pd.DataFrame(data=SK_KNN.scores_, columns=["KNN"], index=[l for l in range((6+add*run)*i, (6+add*run)*(i+1))])


            ### QDA ###
            QDA = QuadraticDiscriminantAnalysis(reg_param=0.5)

            ## Feature selection ## 
            SFS_QDA = SequentialFeatureSelector(QDA, n_features_to_select=n, cv=10)
            SFS_QDA.fit(x,y)
            #x_QDA_val = SFS_QDA.transform(x_val)
            #QDA.fit(x_QDA, y)

            QDA_features = pd.DataFrame(data=SFS_QDA.support_, columns=["QDA"], index=[l for l in range((6+add*run)*i, (6+add*run)*(i+1))])
            time2 = time.time()


            '''
            ### SVC ###
            svc = SVC(kernel="rbf", class_weight="balanced", probability = True)

            ## Features selection ##
            SFS_SVC = SequentialFeatureSelector(svc, n_features_to_select=n, cv=10)
            SFS_SVC.fit(x,y)
            #x_SVC = SFS_SVC.fit_transform(x,y)
            #x_SVC_val = SFS_SVC.transform(x_val)
            #svc.fit(x_SVC, y)
            SVC_features = pd.DataFrame(data=SFS_SVC.support_, columns=["SVC"], index=[l for l in range((6+add*run)*i, (6+add*run)*(i+1))])

            '''
            ### Merging data ###
            feature_scores = feature_scores._append(pd.concat([KNN_features, QDA_features], axis=1))

            

        ### Saving data ###
        if corr == 1:
            feature_scores.to_csv(f"./Data/feature_scores_{n}_feat_extra_feat_corr_{run}", sep=",")
        else:
            feature_scores.to_csv(f"./Data/feature_scores_{n}_feat_extra_feat_{run}", sep=",")

