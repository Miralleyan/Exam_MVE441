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
transform = False #Tries different ways of preprocessing the data, scaling pca, and Kbest features selection


exercise1c = True

######################################################################################################################
##this code classfied using 6 different classifiers

if exercise1c == 1:
    ### Adding extra features ###
    add = 200
    for run in range(1,11):
        print(f"Now have {add*run} features")
        new_features = pd.concat([pd.DataFrame(data = stats.norm(loc =stats.norm(scale = 4).rvs(), scale = 3).rvs(size = len(x_train)), index=x_train.index , columns=[f"S_{run*add+i}"]) for i in range(add)], axis=1)
        x_train = pd.concat([x_train,new_features ], axis = 1)
        
        if run == 1:
            continue

        ### Scaling data ###
        scaler = StandardScaler()
        x_scaled_np = scaler.fit_transform(x_train)
        x_scaled = pd.DataFrame(data = x_scaled_np, index=x_train.index, columns= x_train.columns)


        """
        #plott(x_scaled, 6)
        x_classes = [x_scaled]
        for i in range(7):
            b = [l for l in range(len(x_scaled)) if y_train[l]==i]
            x_classes.append(x_scaled.iloc[b])
        y_classes = [y_train]
        for i in range(7):
            y_classes.append(y_train[y_train == i])
        #print(y_classes)
        """


        '''
        for ind in range(len(x_classes)):
            ind += 1
            print(f"Class{ind}:")
            x_class = x_classes[ind]
            y_class = y_classes[ind]
        '''


        for n in range(1,2):
            n=6
            print(f"Features {n}:")
            con_mat = pd.DataFrame()
            y_pred_mat = pd.DataFrame()
            feature_scores = pd.DataFrame()
            class_prob = pd.DataFrame()

            skf = StratifiedKFold(n_splits=10)
            for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
                print(f"Outer fold {i}:")
                x = x_train.iloc[train_index]
                x_val = x_train.iloc[test_index]
                y = y_train[train_index]
                y_val = y_train[test_index]
                        
                
                ### KNN ###
                KNN = KNeighborsClassifier(n_neighbors=5)

                ## Features selection ##
                SK_KNN = SelectKBest(f_classif, k=n)
                x_KNN = SK_KNN.fit_transform(x,y)
                x_KNN_val = SK_KNN.transform(x_val)
                KNN.fit(x_KNN, y)

                ## Prediction ##
                y_pred = KNN.predict(x_KNN_val)
                KNN_y_pred = pd.DataFrame(data = np.array([y_val, y_pred]).T, columns=["y_val", "KNN_pred"], index=x_val.index)

                if n == 6:
                    KNN_prob = pd.DataFrame(data=KNN.predict_proba(x_KNN_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)
                    
                if exercise1c == 0:
                    KNN_features = pd.DataFrame(data=SK_KNN.scores_, columns=["KNN"], index=[l for l in range(6*i, 6*(i+1))])
                    KNN_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"KNN_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])
                else:
                    KNN_features = pd.DataFrame(data=SK_KNN.scores_, columns=["KNN"], index=[l for l in range((6+add*run)*i, (6+add*run)*(i+1))])


                ### QDA ###
                if exercise1c == 0:
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

                        QDA_features = pd.DataFrame(data=SFS_QDA.support_, columns=["QDA"], index=[l for l in range(6*i, 6*(i+1))])

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
                #print(RFE_LR.support_)
                LR.fit(x_LR, y)

                ## Prediction ##
                y_pred = LR.predict(x_LR_val)
                LR_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["LR_pred"], index= x_val.index)

                if n == 6:
                    LR_prob = pd.DataFrame(data=LR.predict_proba(x_LR_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)

                if exercise1c == 0:
                    LR_features = pd.DataFrame(data=RFE_LR.support_, columns=["LR"], index=[l for l in range(6*i, 6*(i+1))])
                    LR_con_mat = pd.DataFrame(data=confusion_matrix(LR.predict(x_LR_val), y_val), columns=[f"LR_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])

                else:
                    LR_features = pd.DataFrame(data=RFE_LR.support_, columns=["LR"], index=[l for l in range((6+add*run)*i, (6+add*run)*(i+1))])




                ### RF ###
                RF = RandomForestClassifier(n_estimators=500)

                ## Feature selection ##
                RFE_RF = RFE(RF, n_features_to_select=n).fit(x,y)
                x_RF = RFE_RF.transform(x)
                x_RF_val = RFE_RF.transform(x_val)
                #print(RFE_RF.support_)
                RF.fit(x_RF, y)

                ## Predicition ##
                y_pred = RF.predict(x_RF_val)
                RF_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["RF_pred"], index= x_val.index)


                if n == 6:
                    RF_prob = pd.DataFrame(data=RF.predict_proba(x_RF_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)

                if exercise1c == 0:
                    RF_features = pd.DataFrame(data=RFE_RF.support_, columns=["RF"], index=[l for l in range(6*i, 6*(i+1))])
                    RF_con_mat = pd.DataFrame(data=confusion_matrix(RF.predict(x_RF_val), y_val), columns=[f"RF_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])

                else:
                    RF_features = pd.DataFrame(data=RFE_RF.support_, columns=["RF"], index=[l for l in range((6+add*run)*i, (6+add*run)*(i+1))])


                if exercise1c == 0:
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
                    #print(SFS_SVC.support_)

                    ## Prediction ##
                    y_pred = svc.predict(x_SVC_val)
                    SVC_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["SVC_pred"], index= x_val.index)

                    SVC_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"SVC_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])
                    #print("SVC", SVC_index)


                    ### LDA ###
                    LDA = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")

                    ## Feature selection ##
                    RFE_LDA = RFE(LDA, n_features_to_select=n).fit(x,y)
                    x_LDA = RFE_LDA.transform(x)
                    x_LDA_val = RFE_LDA.transform(x_val)
                    #print(RFE_LDA.support_)
                    LDA.fit(x_LDA, y)

                    if n == 6:
                        LDA_prob = pd.DataFrame(data=LDA.predict_proba(x_LDA_val), columns=[f"{fishes[i]}" for i in range(7)], index=x_val.index)
                    
                    LDA_features = pd.DataFrame(data=RFE_LDA.support_, columns=["LDA"], index=[l for l in range(6*i, 6*(i+1))])


                    ## Prediction ##
                    y_pred = LDA.predict(x_LDA_val)
                    LDA_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["LDA_pred"], index= x_val.index)

                    LDA_con_mat = pd.DataFrame(data=confusion_matrix(y_pred, y_val), columns=[f"LDA_{p+1}" for p in range(7)], index=[l for l in range(7*i, 7*(i+1))])
                    #print("LDA", LDA_index)


                ### Merging data ###
                if exercise1c == 0:
                    feature_scores = feature_scores._append(pd.concat([KNN_features, QDA_features, LR_features, RF_features, SVC_features, LDA_features], axis=1))
                    #print(feature_scores)
                    y_pred_mat = y_pred_mat._append(pd.concat([KNN_y_pred, QDA_y_pred, LR_y_pred, RF_y_pred, SVC_y_pred, LDA_y_pred], axis=1))
                    #print(y_pred_mat)
                    con_mat = con_mat._append(pd.concat([KNN_con_mat, QDA_con_mat, LR_con_mat, RF_con_mat, SVC_con_mat, LDA_con_mat], axis=1))
                    #print(result)
                else:
                    feature_scores = feature_scores._append(pd.concat([KNN_features, LR_features, RF_features], axis=1))
                    #print(feature_scores)
                    y_pred_mat = y_pred_mat._append(pd.concat([KNN_y_pred,  LR_y_pred, RF_y_pred], axis=1))
                    #print(y_pred_mat)
                if n == 6 and exercise1c == 0:
                    class_prob = class_prob._append((KNN_prob+ QDA_prob+LR_prob+RF_prob+SVC_prob+LDA_prob)/6)
                    #print(class_prob)

            ### Saving data ###
            if exercise1c == 0:
                feature_scores.to_csv(f"./Data/feature_scores_{n}_feat", sep=",")
                y_pred_mat.to_csv(f"./Data/y_pred_mat_{n}_feat", sep=",")
                con_mat.to_csv(f"./Data/con_mat_{n}_feat", sep=",")
            else:
                feature_scores.to_csv(f"./Data/feature_scores_{n}_feat_extra_feat_{run}", sep=",")
                y_pred_mat.to_csv(f"./Data/y_pred_mat_{n}_feat_extra_feat_{run}", sep=",")
            if n == 6:
                class_prob.to_csv(f"./Data/class_prob_mean", sep=",")




######################################################################################################################
##The following was used to see how I wanted to transform the data, 

if transform == 1:
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
            





