from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest,  SequentialFeatureSelector, f_classif
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from scipy import stats
import numpy as np


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

#If the added features are correlated or not
corr = True

######################################################################################################################
##this code classfied using 6 different classifiers

x_load = pd.read_csv("./x_train_corr", index_col= 0)
### Adding extra features ###
add = 200
for run in range(1,16):
    print(f"Now have {add*run} features")

    ## Not correlated ##
    #new_features = pd.concat([pd.DataFrame(data = stats.norm(loc =stats.norm(scale = 4).rvs(), scale = 3).rvs(size = len(x_train)), index=x_train.index , columns=[f"S_{(run-1)*add+i}"]) for i in range(add)], axis=1)

    ## Correlated ##
    #new_features_corr = pd.concat([pd.DataFrame(data = x_train[x_train.columns[i%6]].to_numpy()*stats.norm(scale = 2).rvs() +stats.norm(scale = 10).rvs(size = len(x_train)), index=x_train.index , columns=[f"S_{(run-1)*add+i}"]) for i in range(add)], axis=1)

    '''
    if corr == 1:
        x_train = pd.concat([x_train,new_features_corr], axis = 1)
        x_train.to_csv("./x_train_corr", sep=",")
    else:
        x_train = pd.concat([x_train, new_features], axis = 1)
        x_train.to_csv("./x_train", sep=",")
    '''
    x_train = x_load[x_load.columns[:200*run+6]]


    ### Scaling data ###
    scaler = StandardScaler()
    x_scaled_np = scaler.fit_transform(x_train)
    x_scaled = pd.DataFrame(data = x_scaled_np, index=x_train.index, columns= x_train.columns)


    for n in range(6,7):
        print(f"Features {n}:")
        feature_scores = pd.DataFrame()
        y_pred_mat = pd.DataFrame()


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
            x_KNN = SK_KNN.fit_transform(x,y)
            x_KNN_val = SK_KNN.transform(x_val)
            
            KNN.fit(x_KNN, y)
            y_pred = KNN.predict(x_KNN_val)

            KNN_features = pd.DataFrame(data=SK_KNN.scores_, columns=["KNN"], index=[l for l in range((6+add*run)*i, (6+add*run)*(i+1))])
            KNN_y_pred = pd.DataFrame(data = np.array([y_val, y_pred]).T, columns=["y_val", "KNN_pred"], index=x_val.index)

            ### QDA ###
            QDA = QuadraticDiscriminantAnalysis(reg_param=0.5)

            ## Feature selection ## 
            SFS_QDA = SequentialFeatureSelector(QDA, n_features_to_select=n, cv=10)
            x_QDA = SFS_QDA.fit_transform(x,y)
            x_QDA_val = SFS_QDA.transform(x_val)

            QDA.fit(x_QDA, y)
            y_pred = QDA.predict(x_QDA_val)


            QDA_features = pd.DataFrame(data=SFS_QDA.support_, columns=["QDA"], index=[l for l in range((6+add*run)*i, (6+add*run)*(i+1))])
            QDA_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["QDA_pred"], index=x_val.index)

            ### Merging data ###
            feature_scores = feature_scores._append(pd.concat([KNN_features, QDA_features], axis=1))
            y_pred_mat = y_pred_mat._append(pd.concat([KNN_y_pred, QDA_y_pred], axis=1))


        ### Saving data ###
        if corr == 1:
            feature_scores.to_csv(f"./Data/feature_scores_{n}_feat_extra_feat_corr_{run}", sep=",")
            y_pred_mat.to_csv(f"./Data/pred_feat_{n}_extra_feat_{run}_corr")
        else:
            feature_scores.to_csv(f"./Data/feature_scores_{n}_feat_extra_feat_{run}", sep=",")
            y_pred_mat.to_csv(f"./Data/pred_feat_{n}_extra_feat_{run}")

