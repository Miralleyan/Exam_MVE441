import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

fish_df = pd.read_csv("./Fish3.txt", sep=" ")

labenc = LabelEncoder()
labenc.fit(fish_df["Species"])
fishes = labenc.classes_
fish_label = labenc.transform(fish_df["Species"])
fish_df = fish_df.drop(labels="Species" , axis=1)

for i in range(7):
    print(f"There are {sum(fish_label==i)} {fishes[i]}")

data = pd.read_csv("./con_mat", sep=",", index_col=0)

#importance = pd.read_csv("./importance", sep=",", index_col=0)
#print(importance)
#importance[importance < 0] = 0


#class_importance = pd.read_csv("./class_importance", sep=",", index_col=0)
#class_importance[class_importance < 0] = 0

methods = ["KNN", "QDA", "LR", "RF", "SVC", "LDA"]


calculate = 1 #Required for alterntives below
total = 0
acc = 0
acc_classes = 1

'''
importance_mat = np.zeros((6,6,10))
std_importance = np.zeros((6,6))
mean_importance = np.zeros((6,6))
for i in range(10):
    for j in range(6):
            mean_importance[:,j] += importance.iloc[[l for l in range(6*i, 6*(i+1))],j]/10
            importance_mat[:, j,i] = importance.iloc[[l for l in range(6*i, 6*(i+1))],j]
   
for i in range(6):
    for j in range(6):
        std_importance[i,j] = importance_mat[i,j,:].std()
#print(mean_importance)
#print(std_importance)


class_importance_mat = np.zeros((6, 42, 10))
std_class_importance = np.zeros((6,42))
mean_class_importance = np.zeros((6,42))
for i in range(10):
    for j in range(42):
        mean_class_importance[:,j] += class_importance.iloc[[l for l in range(6*i, 6*(i+1))],j]/10
        class_importance_mat[:,j,i] = class_importance.iloc[[l for l in range(6*i, 6*(i+1))],j]

for i in range(6):
    for j in range(42):
        std_class_importance[i,j] = class_importance_mat[i,j,:].std()
print(mean_class_importance)
for i in range(6):
    print(mean_class_importance[:,i])
#print(std_class_importance)

'''





if calculate == 1:
    accuracy_1 = np.zeros((10,6))
    class_accuracy = np.zeros((10,6,7))
    class_specificty = np.zeros((10,6,7))
    class_sensitivity = np.zeros((10,6,7))

    for i in range(10):
        for j in range(6):
            con_mat = data.iloc[[l for l in range(7*i,7*(i+1))], [l for l in range(7*j, 7*(j+1))]]
            #print(con_mat)
            #print(sum(con_mat.iloc[k,l] for k in range(7) for l in range(7)))
            for l in range(7):
                accuracy_1[i,j] += con_mat.iloc[l,l]/sum(con_mat.iloc[k,l] for k in range(7) for l in range(7)) ##number of data points
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

    accuracy = np.zeros((10,6))
    for i in range(7):
        accuracy += class_accuracy[:,:,i]/7

    #print(accuracy)
    #print(sensitivity[:,:,6])
    #print(specificty[:,:,6])

if acc*calculate == 1:
    fig, axs = plt.subplots(1,2)

    axs[0].boxplot(accuracy)
    axs[0].title.set_text("Accuracy_mean")
    axs[0].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel='Accuracy' )

    axs[1].boxplot(accuracy_1)
    axs[1].title.set_text("Accuracy")
    axs[1].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel ="Accuracy")
    plt.show()


if acc_classes*calculate == 1:
    fig, axs = plt.subplots(1,7)
    for i in range(7):
        axs[i].boxplot(class_accuracy[:,:,i])
        axs[i].title.set_text(f"{fishes[i]}")
        axs[i].set(xlabel='KNN   QDA  LR   RF  SVC LDA', ylabel='Accuracy', ylim =[0,1] )
    plt.show()

if total*calculate == 1 :
    fig, axs = plt.subplots(1,3)

    axs[0].boxplot(accuracy_1)
    axs[0].title.set_text("Accuracy")
    axs[0].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel='Accuracy' )

    axs[1].boxplot(specificty)
    axs[1].title.set_text("Specificity")
    axs[1].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel ="Specificity")
                
    axs[2].boxplot(sensitivity)
    axs[2].title.set_text("Sensitivity")
    axs[2].set(xlabel='KNN         QDA            LR            RF            SVC          LDA', ylabel = "Sensitivity")
    plt.show()
