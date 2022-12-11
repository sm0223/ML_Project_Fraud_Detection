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
pd.set_option('display.max_columns',None)


##Loading DataSet from Kaggle
x=pd.read_csv("/kaggle/input/preprocessed/train_mod.csv")
y=pd.read_csv("/kaggle/input/preprocessed/train_target.csv")
test=pd.read_csv("../input/preprocessed/test_preprocessed.csv")

##Importing ImbLearn for OverSampling(SMOTE) and UnderSampling(RandomUnderSampler)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

##UnderSampling Performed
rus=RandomUnderSampler(random_state=0)
smote=SMOTE(sampling_strategy='minority')
X_sm,y_sm=rus.fit_resample(x,y)

##Spliting data for Test and Train
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test= train_test_split(X_sm,y_sm,test_size=0.2,stratify=y_sm, random_state=1)

##Parameters for Tuning
params={
        "n_estimators": [20,60,10,50,100],
        "max_depth": [3,10,20,50,None],
        "max_features": [0.1,0.5,0.7,1],
        "max_samples": [0.3,0.5,0.7,1.0]
       }

##Building RandomForest Model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

##Random Search CV for Hyperparameter Tuning
clf=RandomizedSearchCV(rf,param_distributions=params,n_iter=5,scoring='f1_micro',n_jobs=-1,cv=5)
clf.fit(X_train,Y_train)

##Best Parameters found
print(clf.best_params_)
##Best Score
clf.best_score_

##Final Model with Hyperparameters
rf=RandomForestClassifier(n_estimators= 20, max_samples= 1.0,max_features= 0.5, max_depth= 25)

##Training Random Forest
rf.fit(X_train,Y_train)

##Predictions
ypred=rf.predict(X_train)
f1_score(Y_train,ypred)
ypred=rf.predict(X_test)
f1_score(Y_test,ypred)
ypred=rf.predict(test)
pd.DataFrame(ypred).to_csv('rf_pred.csv')
