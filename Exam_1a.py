from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
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



######################################################################################################################
##this code classfied using 6 different classifiers


### Scaling data ###
scaler = StandardScaler()
x_scaled_np = scaler.fit_transform(x_train)
x_scaled = pd.DataFrame(data = x_scaled_np, index=x_train.index, columns= x_train.columns)

y_pred_mat = pd.DataFrame()
con_mat = pd.DataFrame()

skf = StratifiedKFold(n_splits=10)
for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
    print(f"Outer fold {i}:")
    x = x_scaled.iloc[train_index]
    x_val = x_scaled.iloc[test_index]
    y = y_train[train_index]
    y_val = y_train[test_index]


    ### Cross validation ###
    KNN_param = [5, 10, 50, 100]
    QDA_param = [0, 0.1,0.5,0.9, 1]
    LR_param = [0, 0.1,0.5,0.9, 1]
    RF_param = [10,50,100,500]
    SVC_param = ["rbf", "sigmoid", "poly", "linear"]
    LDA_param = [None, "auto", 0.1, 0.5, 0.9]

    KNN_cross_val = np.zeros((10,len(KNN_param)))
    QDA_cross_val = np.zeros((10,len(QDA_param)))
    LR_cross_val = np.zeros((10,len(LR_param)))
    RF_cross_val = np.zeros((10,len(RF_param)))
    SVC_cross_val = np.zeros((10,len(SVC_param)))
    LDA_cross_val = np.zeros((10,len(LDA_param)))

    skf = StratifiedKFold(n_splits=10)
    for j, (train_index, test_index) in enumerate(skf.split(x, y)):
        print(f"Inner fold {j}:")
        
        x_cross = x.iloc[train_index]
        x_cross_val = x.iloc[test_index]
        y_cross = y[train_index]
        y_cross_val = y[test_index]

        ## #KNeighbours, non-linear, non-parametric ###
        for param in range(len(KNN_param)):
            KNN = KNeighborsClassifier(n_neighbors=KNN_param[param])
            KNN.fit(x_cross, y_cross)

            KNN_cross_val[j,param] = f1_score(KNN.predict(x_cross_val), y_cross_val, average="weighted")
        

        ### QDA, non-linear, parametric ###
        for param in range(len(QDA_param)):
            QDA = QuadraticDiscriminantAnalysis(reg_param=QDA_param[param])
            QDA.fit(x_cross, y_cross)

            QDA_cross_val[j,param] = f1_score(QDA.predict(x_cross_val), y_cross_val, average="weighted")

    
        ### LR, linear, parametric ###
        for param in range(len(LR_param)):
            LR = LogisticRegression(penalty="elasticnet", class_weight="balanced", solver="saga", l1_ratio=LR_param[param], max_iter=200)
            LR.fit(x_cross, y_cross)

            LR_cross_val[j,param] = f1_score(LR.predict(x_cross_val), y_cross_val, average="weighted")

        ### RF, non-linear, non-parametric ###
        for param in range(len(RF_param)):
            RF = RandomForestClassifier(n_estimators=RF_param[param])
            RF.fit(x_cross, y_cross)

            RF_cross_val[j,param] = f1_score(RF.predict(x_cross_val), y_cross_val, average="weighted")

        ### SVC, non-linear, non-parametric ###
        for param in range(len(SVC_param)):
            svc = SVC(kernel=SVC_param[param], class_weight="balanced")
            svc.fit(x_cross, y_cross)
            SVC_cross_val[j,param] = f1_score(svc.predict(x_cross_val), y_cross_val, average="weighted")


        ### LDA, linear, parametric ###
        for param in range(len(LDA_param)):
            LDA = LinearDiscriminantAnalysis(solver="eigen", shrinkage=LDA_param[param])
            LDA.fit(x_cross, y_cross)
            LDA_cross_val[j,param] = f1_score(LDA.predict(x_cross_val), y_cross_val, average="weighted")
                   
    
    ### KNN ###
    KNN_index = np.argmax(np.array([np.mean(KNN_cross_val[:,p]) for p in range(len(KNN_param))]))
    KNN = KNeighborsClassifier(n_neighbors=KNN_param[KNN_index])
    KNN.fit(x, y)

    ## Prediction ##
    y_pred = KNN.predict(x_val)
    KNN_y_pred = pd.DataFrame(data = np.array([y_val, y_pred]).T, columns=["y_val", "KNN_pred"], index=x_val.index)
    KNN_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"KNN_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])


    ### QDA ###
    QDA_index = np.argmax(np.array([np.mean(QDA_cross_val[:,p]) for p in range(len(QDA_param))]))
    QDA = QuadraticDiscriminantAnalysis(reg_param=QDA_param[QDA_index])
    QDA.fit(x,y)

    ## Prediction ##
    y_pred = QDA.predict(x_val)
    QDA_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["QDA_pred"], index= x_val.index)
    QDA_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"QDA_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])



    ### LR ###
    LR_index = np.argmax(np.array([np.mean(LR_cross_val[:,p]) for p in range(len(LR_param))]))
    LR = LogisticRegression(penalty="elasticnet", class_weight="balanced", solver="saga", l1_ratio=LR_param[LR_index], max_iter=200)
    LR.fit(x, y)

    ## Prediction ##
    y_pred = LR.predict(x_val)
    LR_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["LR_pred"], index= x_val.index)
    LR_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"LR_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])


    ### RF ###
    RF_index = np.argmax(np.array([np.mean(RF_cross_val[:,p]) for p in range(len(RF_param))]))
    RF = RandomForestClassifier(n_estimators=RF_param[RF_index])
    RF.fit(x, y)

    ## Predicition ##
    y_pred = RF.predict(x_val)
    RF_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["RF_pred"], index= x_val.index)
    RF_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"RF_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])



    ### SVC ###
    SVC_index = np.argmax(np.array([np.mean(SVC_cross_val[:,p]) for p in range(len(SVC_param))]))
    svc = SVC(kernel=SVC_param[SVC_index], class_weight="balanced")
    svc.fit(x,y)

    ## Prediction ##
    y_pred = svc.predict(x_val)
    SVC_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["SVC_pred"], index= x_val.index)
    SVC_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"SVC_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])



    ### LDA ###
    LDA_index = np.argmax(np.array([np.mean(LDA_cross_val[:,p]) for p in range(len(LDA_param))]))
    LDA = LinearDiscriminantAnalysis(solver="eigen", shrinkage=LDA_param[LDA_index])
    LDA.fit(x, y)

    ## Prediction ##
    y_pred = LDA.predict(x_val)
    LDA_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["LDA_pred"], index= x_val.index)
    LDA_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"LDA_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])


    ### Merging data ###
    y_pred_mat = y_pred_mat._append(pd.concat([KNN_y_pred, QDA_y_pred, LR_y_pred, RF_y_pred, SVC_y_pred, LDA_y_pred], axis=1))
    con_mat = con_mat._append(pd.concat([KNN_con_mat, QDA_con_mat, LR_con_mat, RF_con_mat, SVC_con_mat, LDA_con_mat], axis=1))


### Saving data ###
y_pred_mat.to_csv(f"./Data/y_pred_mat", sep=",")
con_mat.to_csv(f"./Data/con_mat", sep=",")

    
