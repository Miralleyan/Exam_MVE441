import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
### Load data ###
fish_df = pd.read_csv("./Fish3.txt", sep=" ")
colors = ["Blue", "Red", "Yellow", "Green", "Purple", "Black", "Pink"]

## Modify it to easier form
labenc = LabelEncoder()
labenc.fit(fish_df["Species"])
fishes = labenc.classes_
fish_label = labenc.transform(fish_df["Species"])
fish_df = fish_df.drop(labels="Species" , axis=1)

## Split test data for later
x_train, x_test, y_train, y_test = train_test_split(fish_df, fish_label, train_size=0.7,random_state=42)

features = list(x_train.columns)

### Scaling data ###
scaler = StandardScaler()
x_scaled_np = scaler.fit_transform(x_train)
x_scaled = pd.DataFrame(data = x_scaled_np, index=x_train.index, columns= x_train.columns)

def plott(x, k):
    fig, axs = plt.subplots(k,k)
    for j in range(k):
        for l in range(k):
            for color, i, species in zip(colors, range(7), fishes):
                axs[j,l].scatter(x[y_train == i, j],x[y_train == i, l], color = color, label = species)
                #plt.title(f"PCA {j} and {k}")
    plt.legend()
    plt.show()
#plott(x_scaled.to_numpy(),6)

for i in range(7):
    print(f"There are {sum(fish_label==i)} {fishes[i]}")

methods = ["KNN", "QDA", "LR", "RF", "SVC", "LDA"]



def accuracy_fcn(data):
    accuracy = np.zeros((10,6))
    class_accuracy = np.zeros((10,6,7))

    for i in range(10):
        for j in range(6):
            con_mat = data.iloc[[l for l in range(7*i,7*(i+1))], [l for l in range(7*j, 7*(j+1))]]
            #print(con_mat)
            #print(sum(con_mat.iloc[k,l] for k in range(7) for l in range(7)))
            for l in range(7):
                accuracy[i,j] += con_mat.iloc[l,l]/sum(con_mat.iloc[k,q] for k in range(7) for q in range(7)) ##number of data points
                denom = 0
                for s in range(7):
                    denom += con_mat.iloc[s,s]
                    class_accuracy[i,j,l] += con_mat.iloc[s,s]
                denom += sum(con_mat.iloc[l, k] for k in range(7) if l != k) + sum(con_mat.iloc[k,l] for k in range(7) if l != k)
                class_accuracy[i,j,l] = class_accuracy[i,j,l]/denom


    return accuracy, class_accuracy

def calculate(data):
    accuracy = np.zeros((10,6))
    class_accuracy = np.zeros((10,6,7))
    class_specificty = np.zeros((10,6,7))
    class_sensitivity = np.zeros((10,6,7))

    for i in range(10):
        for j in range(6):
            con_mat = data.iloc[[l for l in range(7*i,7*(i+1))], [l for l in range(7*j, 7*(j+1))]]
            print(con_mat)
            #print(sum(con_mat.iloc[k,l] for k in range(7) for l in range(7)))
            for l in range(7):
                accuracy[i,j] += con_mat.iloc[l,l]/sum(con_mat.iloc[k,l] for k in range(7) for l in range(7)) ##number of data points
                class_accuracy[i,j,l] = con_mat.iloc[l,l]/sum(con_mat.iloc[k,l] for k in range(0,7))
                
                denom_spec = [con_mat.iloc[p,s] for p in range(7) for s in range(7) if s!=l]
                class_specificty[i,j,l] = sum([con_mat.iloc[p,s] for p in range(7) for s in range(7) if s!=l and p!=l])/sum(denom_spec)

                denom_sen = [con_mat.iloc[p,l] for p in range(7)]
                class_sensitivity[i,j,l] = con_mat.iloc[l,l]/sum(denom_sen)
    sensitivity = np.zeros((10,6))
    for i in range(7):
        sensitivity += class_sensitivity[:,:,i]/7
    #print(sensitivity)

    specificty = np.zeros((10,6))
    for i in range(7):
        specificty += class_specificty[:,:,i]/7
    #print(specificty)

    accuracy_mean = np.zeros((10,6))
    for i in range(7):
        accuracy_mean += class_accuracy[:,:,i]/7
    return sensitivity, specificty, accuracy, class_sensitivity, class_specificty, class_accuracy

extra_feat = 1
plot_certain = 0
plot_uncertain = 0
plot_mislabel  = 0
all_feat = 0
calc = 0
total = 0
acc_classes = 0
feat_dic = 0
plot_accuracy_feat = 0
acc_classes_features = 0

if extra_feat == 1:
    KNN_acc = np.zeros((11,1))
    QDA_acc = np.zeros((11,1))
    for i in range(1,11):
        y_pred = pd.read_csv(f"./Data/y_pred_mat_6_feat_extra_feat_{i}", index_col=0)
        KNN_acc[i] = sum(y_pred["KNN_pred"] == y_pred["y_val"])/len(y_pred["y_val"])
        QDA_acc[i] = sum(y_pred["QDA_pred"] == y_pred["y_val"])/len(y_pred["y_val"])
    print(KNN_acc)
    print(QDA_acc)


if plot_certain == 1:
    data = pd.read_csv(f"./Data/y_pred_mat_6_feat", sep=",", index_col=0)
    prob = pd.read_csv("./Data/class_prob_mean", sep=",", index_col=0)
    print(prob)

    pred_dic = {ind: 0 for ind in data.index}
    label_dic = {ind: [0,0,0,0,0,0,0] for ind in data.index}

    for ind in data.index:
        pred = data.loc[ind].values
        #print(pred)
        for i in range(1,7):
            if pred[0] == pred[i]:
                pred_dic[ind] += 1
            else:
                label_dic[ind][pred[i]] += 1 


    #cert_dic = {ind:np.argmax(prob.loc[ind]) for ind in prob.index if (sum(prob.loc[ind]>0.5)>0 and sum(prob.loc[ind]>0.20) < 2)and np.argmax(prob.loc[ind]) == data.loc[ind].values[0] }
    cert_dic = {ind:np.argmax(prob.loc[ind]) for ind in prob.index if (sum(prob.loc[ind]>0.7)>0)and np.argmax(prob.loc[ind]) == data.loc[ind].values[0] }

    #print([prob.loc[ind] for ind in prob_dic])
    print(len(cert_dic))


    certain = x_scaled.loc[[x for x in cert_dic]].to_numpy()
    j = 0
    l= 4
    for color, i, species in zip(colors, range(7), fishes):
        plt.scatter(x_scaled_np[y_train == i, j],x_scaled_np[y_train == i, l], color = color, label = species)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 5")
    plt.scatter(certain[:,j], certain[:,l], c = "cyan", label ="Certain")
    plt.legend()
    plt.show()

if plot_uncertain == 1:
    prob = pd.read_csv("./Data/class_prob_mean", sep=",", index_col=0)
    #print(prob)
    prob_dic = {ind:np.argmax(prob.loc[ind]) for ind in prob.index if (sum(prob.loc[ind]<0.5)== 7 and sum(prob.loc[ind]>0.35) >=2)}
    #print([prob.loc[ind] for ind in prob_dic])
    print(len(prob_dic))

    print({ind for ind in cert_dic if ind in prob_dic})

    uncertain = x_scaled.loc[[x for x in prob_dic]].to_numpy()
    j = 0
    l= 4
    #for color, i, species in zip(colors, range(7), fishes):
    #    plt.scatter(x_scaled_np[y_train == i, j],x_scaled_np[y_train == i, l], color = color, label = species)
    #    plt.xlabel("Feature 1")
    #    plt.ylabel("Feature 5")
    plt.scatter(uncertain[:,j], uncertain[:,l], c = "orange", label ="Uncertain?")
    #plt.legend()
    #plt.show()

if plot_mislabel == 1:
    
    data = pd.read_csv(f"./Data/y_pred_mat_6_feat", sep=",", index_col=0)
    prob = pd.read_csv("./Data/class_prob_mean", sep=",", index_col=0)
    #print(prob)
    prob_dic = {ind:np.argmax(prob.loc[ind]) for ind in prob.index if sum(prob.loc[ind]>=0.7) >0}

    pred_dic = {ind: 0 for ind in data.index}
    label_dic = {ind: [0,0,0,0,0,0,0] for ind in data.index}

    for ind in data.index:
        pred = data.loc[ind].values
        #print(pred)
        for i in range(1,7):
            if pred[0] == pred[i]:
                pred_dic[ind] += 1
            else:
                label_dic[ind][pred[i]] += 1 



    label_dic_6 = {x[0]:x[1].index(6) for x in label_dic.items() if (6 in x[1])}
    label_dic_5 = {x[0]:x[1].index(5) for x in label_dic.items() if (5 in x[1])}
    label_dic_4 = {x[0]:x[1].index(4) for x in label_dic.items() if (4 in x[1])}

    ## Remove for harder criteria
    label_dic_6.update(label_dic_5)
    #label_dic_6.update(label_dic_4)

    #dic = {x[0]:x[1] for x in pred_dic.items() if (x[1] == 0 or x[1] == 1) or x[1]==2}
    dic = {x[0]:x[1] for x in pred_dic.items() if (x[1] == 0 or x[1] == 1)}
    #dic = {x[0]:x[1] for x in pred_dic.items() if (x[1] == 0)}

    mislabel = {x[0]:x[1] for x in label_dic_6.items() if (x[0] in dic and x[0] in prob_dic and x[1]==prob_dic[x[0]]) }
    print(len(mislabel))

    #print(mislabel)

    #print([x[0] for x in pred_dic.items() if x[1] == 0])

    failed = x_scaled.loc[[x for x in dic]].to_numpy()
    mislab = x_scaled.loc[[x for x in mislabel]].to_numpy()
    col = [colors[x[1]] for x in mislabel.items()]
    #print(x_train.iloc[16])
    og_col = [colors[y_train[x_train.index.get_loc(x)]] for x in mislabel]
    #print(mislabel)
    #for lab in mislabel:
    #    print(mislabel[lab])
    #    print(lab, prob.loc[lab])

    k=6

    #fig, axs = plt.subplots(k,k)
    j = 0
    l= 4
    for color, i, species in zip(colors, range(7), fishes):
        plt.scatter(x_scaled_np[y_train == i, j],x_scaled_np[y_train == i, l], color = color, label = species)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 5")
    #plt.scatter(failed[:,j], failed[:,l], color = "orange" , label ="Mislabel?", s = 100)
    plt.legend()
    plt.scatter(mislab[:,j], mislab[:,l], c = og_col, label ="Mislabel?", s = 150)
    plt.scatter(mislab[:,j], mislab[:,l],c = col, label ="Mislabel?", s = 25)
    plt.show()

if all_feat == 1:
    k=6
    fig, axs = plt.subplots(k,k)
    for j in range(k):
        for l in range(k):
            for color, i, species in zip(colors, range(7), fishes):
                axs[j,l].scatter(x_scaled_np[y_train == i, j],x_scaled_np[y_train == i, l], color = color, label = species)
                #axs[j,l].set_ylabel(f"Feature {l}")
            axs[j,l]. set_xticklabels([]) 
            axs[j,l].set_title(f"{features[j]} vs {features[l]}", fontsize =10, pad =5)

    plt.legend()
    plt.show()

if acc_classes_features== 1:
    for j in range(7):

        fig, axs = plt.subplots(1,6)
        for i in range(6):
            data = pd.read_csv(f"./Data/con_mat_{i+1}_feat", sep=",", index_col=0)
            accuracy, class_accuracy = accuracy_fcn(data)
            axs[i].boxplot(class_accuracy[:,:,j])
            axs[i].title.set_text(f"{i+1} features")
            axs[i].set(xlabel='KNN   QDA  LR   RF  SVC LDA', ylabel='Accuracy', ylim =[0,1] )
        plt.show()


if feat_dic == 1:
    for s in range(1,7):
        feature_dict = {model: [0]*6 for model in methods}
        #print(feature_dict)
        feature_scores = pd.read_csv(f"./Data/feature_scores_{s}_feat", index_col=0)

        new = np.zeros((60,1))
        #new_QDA = np.zeros((60,1))
        for t in range(10):
            feat_score = feature_scores.iloc[[p for p in range(6*t, 6*(t+1))]]["KNN"].values
            #feat_score_QDA = feature_scores.iloc[[p for p in range(6*t, 6*(t+1))]]["QDA"].values
            #print(feat_score)
            
            for f in range(s):
                ind = np.argmax(feat_score)
                #ind_QDA = np.argmax(feat_score_QDA)
                new[6*t+ind] = 1
                #new_QDA[6*t+ind_QDA] = True

                feat_score[ind] = -1000
                #feat_score_QDA[ind] = -1000
        #feature_scores.QDA = pd.DataFrame(new_QDA, columns=["QDA"])
        feature_scores.KNN = pd.DataFrame(new, columns=["KNN"])
        #print(feature_scores)



        for model in methods:
            for i in range(10):

                feature_dict[model]+= feature_scores.iloc[[p for p in range(6*i, 6*(i+1))]][model].values
        print(feature_dict)
            




if plot_accuracy_feat == 1:
    mean_feature_accuracy = np.zeros((6,6)) #Mean accuracy for all models for all number of features, columns features, rows models
    mean_accuracy = np.zeros((6,1)) #Mean accuracy of all models for each number of features
    fig, axs = plt.subplots(1,6)
    for i in range(6):
        data = pd.read_csv(f"./Data/con_mat_{(i)+1}_feat", sep=",", index_col=0)

        accuracy, class_accuracy = accuracy_fcn(data)

        for s in range(6):
            mean_accuracy[s] += accuracy[:,s].mean()/6
            #print(mean_feature_accuracy[s,:])
            mean_feature_accuracy[s,i] = accuracy[:,s].mean()
            #print(mean_feature_accuracy)
        #print(accuracy)


        axs[i].boxplot(accuracy)
        axs[i].title.set_text(f"{i+1} features")
        axs[i].set(xlabel='KNN   QDA  LR   RF  SVC LDA', ylabel='Accuracy', ylim =[0.2, 1] )
    plt.show()

    #print(accuracy)
    for i in range(6):
        plt.plot([1,2,3,4,5,6],mean_feature_accuracy[i,:], label = methods[i])
    plt.xlabel("Number of features")
    plt.ylabel("Mean accuracy for model over 10 runs")
    plt.ylim([0.3,1])
    plt.legend()
    plt.show()




if calc == 1:
    sensitivity, specificty, accuracy, class_sensitivity, class_specificty, class_accuracy = calculate(data)
    print(accuracy)

if acc_classes== 1:
    data = pd.read_csv("./Data/con_mat_6_feat", sep=",", index_col=0)
    pred = pd.read_csv("./Data/y_pred_mat_6_feat", sep=",", index_col=0)
    #print(pred)
    #print(data[[f"SVC_{i}" for i in range(1,8)]])
    #print(data[[f"RF_{i}" for i in range(1,8)]])
    accuracy, class_accuracy = accuracy_fcn(data)
    fig, axs = plt.subplots(1,7)
    for i in range(7):
        axs[i].boxplot(class_accuracy[:,:,i], showmeans = True, meanprops = {"marker":"*"})
        axs[i].title.set_text(f"{fishes[i]}")
        axs[i].set_xticks(list(range(1,7)), labels = ["KNN", "QDA", "LR", "RF","SVC", "LDA"])
        axs[i].set( ylabel='Accuracy', ylim =[0.55,1] )
    plt.show()

if total == 1 :
    data = pd.read_csv("./Data/con_mat_6_feat", index_col=0)
    #print(data)
    sensitivity, specificty, accuracy, class_sensitivity, class_specificty, class_accuracy = calculate(data)
    #print(accuracy)
    fig, axs = plt.subplots(1,3)

    axs[0].boxplot(accuracy)
    axs[0].title.set_text("Accuracy")
    axs[0].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel='Accuracy', ylim=[0.4,1] )

    axs[1].boxplot(specificty)
    axs[1].title.set_text("Specificity")
    axs[1].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel ="Specificity", ylim=[0.4,1] )
                
    axs[2].boxplot(sensitivity)
    axs[2].title.set_text("Sensitivity")
    axs[2].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel = "Sensitivity", ylim=[0.4,1] )
    plt.show()

    fig, axs = plt.subplots(1,7)
    for i in range(7):
        axs[i].boxplot(class_sensitivity[:,:,i], showmeans = True, meanprops = {"marker":"*"})
        axs[i].title.set_text(f"{fishes[i]}")
        axs[i].set_xticks(list(range(1,7)), labels = ["KNN", "QDA", "LR", "RF","SVC", "LDA"])
        axs[i].set( ylabel='Sensitvity', ylim =[0,1] )
    plt.show()

    fig, axs = plt.subplots(1,7)
    for i in range(7):
        axs[i].boxplot(class_specificty[:,:,i], showmeans = True, meanprops = {"marker":"*"})
        axs[i].title.set_text(f"{fishes[i]}")
        axs[i].set_xticks(list(range(1,7)), labels = ["KNN", "QDA", "LR", "RF","SVC", "LDA"])
        axs[i].set( ylabel='Specificity', ylim =[0,1] )
    plt.show()

'''

fig, axs = plt.subplots(1,2)

axs[0].boxplot(accuracy_mean)
axs[0].title.set_text("Accuracy_mean")
axs[0].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel='Accuracy' )

axs[1].boxplot(accuracy)
axs[1].title.set_text("Accuracy")
axs[1].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel ="Accuracy")
plt.show()

'''