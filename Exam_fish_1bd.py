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





######################################################################################################################
### Scaling data ###
scaler = StandardScaler()
x_scaled_np = scaler.fit_transform(x_train)
x_scaled = pd.DataFrame(data = x_scaled_np, index=x_train.index, columns= x_train.columns)


for n in range(1,7):
    print(f"Features {n}:")
    con_mat = pd.DataFrame()
    y_pred_mat = pd.DataFrame()
    feature_scores = pd.DataFrame()
    class_prob = pd.DataFrame()

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
        

        ## Prediction ##
        KNN.fit(x_KNN, y)
        y_pred = KNN.predict(x_KNN_val)
        KNN_y_pred = pd.DataFrame(data = np.array([y_val, y_pred]).T, columns=["y_val", "KNN_pred"], index=x_val.index)
        if n == 6:
            KNN_prob = pd.DataFrame(data=KNN.predict_proba(x_KNN_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)
            
        KNN_features = pd.DataFrame(data=SK_KNN.scores_, columns=["KNN"], index=[l for l in range(6*i, 6*(i+1))])
        KNN_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"KNN_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])


        ### QDA ###
        QDA = QuadraticDiscriminantAnalysis(reg_param=0.5)

        ## Feature selection ## 
        if n == 6:
            x_QDA = x
            x_QDA_val = x_val
            QDA.fit(x,y)

            QDA_prob = pd.DataFrame(data=QDA.predict_proba(x_QDA_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)

            QDA_features = pd.DataFrame(data=[True]*6, columns=["QDA"], index=[l for l in range(6*i, 6*(i+1))])
        else:
            SFS_QDA = SequentialFeatureSelector(QDA, n_features_to_select=n, cv=10)
            x_QDA = SFS_QDA.fit_transform(x,y)
            x_QDA_val = SFS_QDA.transform(x_val)
            QDA.fit(x_QDA, y)

            QDA_features = pd.DataFrame(data=SFS_QDA.support_, columns=["QDA"], index=[l for l in range((6)*i, (6)*(i+1))])

        ## Prediction ##
        y_pred = QDA.predict(x_QDA_val)
        QDA_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["QDA_pred"], index= x_val.index)
        #print(QDA_y_pred)
        QDA_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"QDA_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])
         #print("QDA", QDA_index)


        ### LR ###
        LR = LogisticRegression(penalty="elasticnet", class_weight="balanced", solver="saga", l1_ratio=0.5, max_iter=200)

        ## Feature selection ##
        RFE_LR = RFE(LR, n_features_to_select=n).fit(x,y)
        x_LR = RFE_LR.transform(x)
        x_LR_val = RFE_LR.transform(x_val)
        

        ## Prediction ##
        LR.fit(x_LR, y)
        print(LR.coef_)
        y_pred = LR.predict(x_LR_val)

        if n == 6:
            LR_prob = pd.DataFrame(data=LR.predict_proba(x_LR_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)

        ## Save data ##
        LR_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["LR_pred"], index= x_val.index)
        LR_features = pd.DataFrame(data=RFE_LR.support_, columns=["LR"], index=[l for l in range(6*i, 6*(i+1))])
        LR_con_mat = pd.DataFrame(data=confusion_matrix(LR.predict(x_LR_val), y_val), columns=[f"LR_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])



        ### RF ###
        RF = RandomForestClassifier(n_estimators=500)

        ## Feature selection ##
        RFE_RF = RFE(RF, n_features_to_select=n).fit(x,y)
        x_RF = RFE_RF.transform(x)
        x_RF_val = RFE_RF.transform(x_val)
        

        ## Predicition ##
        RF.fit(x_RF, y)
        y_pred = RF.predict(x_RF_val)

        if n == 6:
            RF_prob = pd.DataFrame(data=RF.predict_proba(x_RF_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)

        ## Save data ##
        RF_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["RF_pred"], index= x_val.index)
        RF_features = pd.DataFrame(data=RFE_RF.support_, columns=["RF"], index=[l for l in range(6*i, 6*(i+1))])
        RF_con_mat = pd.DataFrame(data=confusion_matrix(RF.predict(x_RF_val), y_val), columns=[f"RF_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])



        ### SVC ###
        svc = SVC(kernel="rbf", class_weight="balanced", probability = True)

        ## Features selection ##
        if n == 6:
            x_SVC = x
            x_SVC_val = x_val
            svc.fit(x,y)

            SVC_prob = pd.DataFrame(data=svc.predict_proba(x_SVC_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)
            SVC_features = pd.DataFrame(data=[True]*6, columns=["SVC"], index=[l for l in range(6*i, 6*(i+1))])

        else:
            SFS_SVC = SequentialFeatureSelector(svc, n_features_to_select=n, cv=10)
            x_SVC = SFS_SVC.fit_transform(x,y)
            x_SVC_val = SFS_SVC.transform(x_val)
            svc.fit(x_SVC, y)

            SVC_features = pd.DataFrame(data=SFS_SVC.support_, columns=["SVC"], index=[l for l in range(6*i, 6*(i+1))])


        ## Prediction ##
        y_pred = svc.predict(x_SVC_val)

        ## Save data ##
        SVC_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["SVC_pred"], index= x_val.index)
        SVC_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"SVC_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])


        ### LDA ###
        LDA = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")

        ## Feature selection ##
        RFE_LDA = RFE(LDA, n_features_to_select=n).fit(x,y)
        x_LDA = RFE_LDA.transform(x)
        x_LDA_val = RFE_LDA.transform(x_val)
        
        if n == 6:
            LDA_prob = pd.DataFrame(data=LDA.predict_proba(x_LDA_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)
        


        ## Prediction ##
        LDA.fit(x_LDA, y)
        y_pred = LDA.predict(x_LDA_val)

        ## Save data ##
        LDA_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["LDA_pred"], index= x_val.index)
        LDA_features = pd.DataFrame(data=RFE_LDA.support_, columns=["LDA"], index=[l for l in range(6*i, 6*(i+1))])
        LDA_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"LDA_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])
        #print("LDA", LDA_index)


        ### Merging data ###
        feature_scores = feature_scores._append(pd.concat([KNN_features, QDA_features, LR_features, RF_features, SVC_features, LDA_features], axis=1))
        y_pred_mat = y_pred_mat._append(pd.concat([KNN_y_pred, QDA_y_pred, LR_y_pred, RF_y_pred, SVC_y_pred, LDA_y_pred], axis=1))
        con_mat = con_mat._append(pd.concat([KNN_con_mat, QDA_con_mat, LR_con_mat, RF_con_mat, SVC_con_mat, LDA_con_mat], axis=1))

        if n == 6:
            class_prob = class_prob._append((KNN_prob+ QDA_prob+LR_prob+RF_prob+SVC_prob+LDA_prob)/6)


    ### Saving data ###
    feature_scores.to_csv(f"./Data/feature_scores_{n}_feat", sep=",")
    y_pred_mat.to_csv(f"./Data/y_pred_mat_{n}_feat", sep=",")
    con_mat.to_csv(f"./Data/con_mat_{n}_feat", sep=",")

    if n == 6:
        class_prob.to_csv(f"./Data/class_prob_mean", sep=",")


            





