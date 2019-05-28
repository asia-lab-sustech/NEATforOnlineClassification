import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.neural_network import MLPClassifier
from sklearn import metrics #accuracy measure

def accuracy_function(eval,pred):
    
    eval=np.array(eval)
    good_index= np.where(eval==1)
    bad_index=np.where(eval==0)
    accuracy=0
    good_acc=0
    bad_acc=0

    for i in range(len(eval)):
        result=eval[i]

        if result==pred[i]:

            accuracy=accuracy+1

            if result==0:
                bad_acc=bad_acc+1
            else:
                good_acc=good_acc+1
    #print(accuracy,good_acc,bad_acc)
    good_acc=good_acc/len(list(good_index[0])) if len(list(good_index[0])) != 0 else 0
    bad_acc=bad_acc/len(list(bad_index[0])) if len(list(bad_index[0])) != 0 else 0
    return accuracy/len(pred),good_acc,bad_acc


data=pd.read_csv('~//data//Anainfo20172018.csv')
data['issue_d'] = pd.to_datetime(data['issue_d'],errors = 'coerce')
train=data.loc[data['issue_d'].dt.year==2017]
test=data.loc[data['issue_d'].dt.year==2018]
train.drop('issue_d', axis=1, inplace=True)
y_train = train['loan_status']
X_train = train.drop('loan_status', axis=1)

Test_set_x=[]
Test_set_y=[]
for i in range(12):
    current=test.loc[test['issue_d'].dt.month==i+1]
    current.drop('issue_d', axis=1, inplace=True)
    Test_set_y.append(current['loan_status'])
    Test_set_x.append(current.drop('loan_status', axis=1))

xyz=[]
std=[]
G=[]
B=[]
# classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Random Forest','MLP']
# models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),
#         RandomForestClassifier(n_estimators=100),
#        MLPClassifier(hidden_layer_sizes=(5, 5),activation='logistic',solver='adam',alpha=0.001,max_iter=5000)]
# classifiers=['Logistic Regression','KNN','Decision Tree']
# models=[LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier()]
classifiers=['Logistic Regression','KNN','Decision Tree','Random Forest','MLP']
models=[LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
       MLPClassifier(hidden_layer_sizes=(5, 5),activation='logistic',solver='adam',alpha=0.001,max_iter=5000)]
for i,m in enumerate(models):
    model = m
    acc=[]
    good=[]
    bad=[]
    model.fit(X_train,y_train)
    for j in range(12):
        y_pred=model.predict(Test_set_x[j])
        results=accuracy_function(Test_set_y[j],y_pred)
        acc.append(results[0])
        good.append(results[1])
        bad.append(results[2])
    print(classifiers[i])
        
    print(acc)
#new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std,'positive':G,'negative':B},index=classifiers)
#print(new_models_dataframe2)