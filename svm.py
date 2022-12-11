import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV,KFold,RandomizedSearchCV

#Importing the dataset
pd.set_option('display.max_columns',None)
x=pd.read_csv("/kaggle/input/preprocessed/train_mod.csv")
y=pd.read_csv("/kaggle/input/preprocessed/train_target.csv")
test=pd.read_csv("../input/preprocessed/test_preprocessed.csv")

#Imblearn is used for Under and Oversampling of Imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#Under Sampling is done
rus=RandomUnderSampler(random_state=0)
smote=SMOTE(sampling_strategy='minority')
X_sm,y_sm=rus.fit_resample(x,y)

#Splitting data for train and test
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test= train_test_split(X_sm,y_sm,test_size=0.2,stratify=y_sm, random_state=1)

#SVM implementation
from sklearn import svm
sv = svm.SVC()

#Parameters for Hyperparameter Tuning
params={
        "kernel": ['rbf'],
        "gamma": [0.01],
        "C": [2,3,5]
       }
#Hyperparameter tuning using RandomizedSearchCV
clf=RandomizedSearchCV(sv,param_distributions=params,scoring='f1_micro',n_jobs=-1,cv=2)
clf.fit(X_train,Y_train)
print(clf.best_params_)

print(clf.best_score_)

sv = svm.SVC(kernel="rbf",gamma=1,C=0.1)
sv.fit(X_train,Y_train)

f1_score(Y_train,sv.predict(X_train))
f1_score(Y_test,sv.predict(X_test))

sv = svm.SVC(kernel="rbf",gamma=0.1,C=1)
sv.fit(X_train,Y_train)
f1_score(Y_train,sv.predict(X_train))
f1_score(Y_test,sv.predict(X_test))

sv = svm.SVC(kernel="rbf",gamma=1,C=100)
sv.fit(X_train,Y_train)
f1_score(Y_train,sv.predict(X_train))
f1_score(Y_test,sv.predict(X_test))

sv = svm.SVC(kernel="rbf",gamma=1,C=10)
sv.fit(X_train,Y_train)
f1_score(Y_train,sv.predict(X_train))
f1_score(Y_test,sv.predict(X_test))


#Final model with parameters
sv = svm.SVC(kernel="rbf",gamma=0.01,C=3)
sv.fit(X_train,Y_train)
f1_score(Y_train,sv.predict(X_train))

#F1 Score on Test data
f1_score(Y_test,sv.predict(X_test))
ypred=sv.predict(test)
pd.DataFrame(ypred).to_csv('svm_pred.csv')
