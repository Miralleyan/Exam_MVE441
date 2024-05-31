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
print(x_train)
x_train_np = x_train.to_numpy()

# Choose which part of the code to run
transform = False #Tries different ways of preprocessing the data, scaling pca, and Kbest features selection


corr = True

######################################################################################################################
##this code classfied using 6 different classifiers


### Adding extra features ###
add = 200
for run in range(1,16):
    print(f"Now have {add*run} features")

    ## Not correlated ##
    new_features = pd.concat([pd.DataFrame(data = stats.norm(loc = stats.norm(scale = 20).rvs(), scale = 40).rvs(size = len(x_train)), index=x_train.index , columns=[f"S_{run*add+i}"]) for i in range(add)], axis=1)

    ## Correlated ##
    new_features_corr = pd.concat([pd.DataFrame(data = x_train[x_train.columns[i%6]].to_numpy()*stats.norm(scale = 2).rvs() +stats.norm(scale = 10).rvs(size = len(x_train)), index=x_train.index , columns=[f"S_{(run-1)*add+i}"]) for i in range(add)], axis=1)
    
    if corr == 1:
        x_train = pd.concat([x_train,new_features_corr], axis = 1)
    else:
        x_train = pd.concat([x_train,new_features], axis = 1)
    
    #if run == 1:
    #    continue

    ### Scaling data ###
    scaler = StandardScaler()
    x_scaled_np = scaler.fit_transform(x_train)
    x_scaled = pd.DataFrame(data = x_scaled_np, index=x_train.index, columns= x_train.columns)


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
        KNN.fit(x, y)

        ## Prediction ##
        y_pred = KNN.predict(x_val)
        KNN_y_pred = pd.DataFrame(data = np.array([y_val, y_pred]).T, columns=["y_val", "KNN_pred"], index=x_val.index)


        ### QDA ###
        QDA = QuadraticDiscriminantAnalysis(reg_param=0.5)
        QDA.fit(x, y)
        
        ## Prediction ##
        y_pred = QDA.predict(x_val)
        QDA_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["QDA_pred"], index= x_val.index)
        #print(QDA_y_pred)


        ### SVC ###
        svc = SVC(kernel="rbf", class_weight="balanced", probability = True)
        svc.fit(x, y)

        ## Prediction ##
        y_pred = svc.predict(x_val)
        SVC_y_pred = pd.DataFrame(data = np.array([y_pred]).T, columns=["SVC_pred"], index= x_val.index)



        ### Merging data ###
        y_pred_mat = y_pred_mat._append(pd.concat([KNN_y_pred, QDA_y_pred, SVC_y_pred], axis=1))

    ### Saving data ###
    y_pred_mat.to_csv(f"./Data/y_pred_mat_extra_feat_{run}", sep=",")
