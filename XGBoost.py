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

#Loading train and test dataset
x=pd.read_csv("train_mod.csv")
y=pd.read_csv("train_target.csv")
test=pd.read_csv("test_preprocessed.csv")

#Imblearn SMOTE for OverSampling
from imblearn.over_sampling import SMOTE
#Imblearn UnderSampler for Undersampling
from imblearn.under_sampling import RandomUnderSampler

#UnderSampling Performed
rus=RandomUnderSampler(random_state=0)
X_sm,y_sm=rus.fit_resample(x,y)

import xgboost as xgb

#XGboost model
my_model = xgb.XGBClassifier()

#XGboost Hyperparameters for Tuning
params={"learning_rate": [0.1,0.15,0.2,0.25,0.3,0.35,0.5,0.7],
        "max_depth": [3,2,5,6,8,10,12,15,20,30,50],
        "min_child_weight": [1,3,5,7],
        "gamma": [0.0,0.1,0.2,0.3],
        "colsample_bytree": [0.3,0.5,0.7,0.4]
       }


# Random Search CV for HyperParameter Tuning. GridSearch CV may be better but Random SearchCV is Faster with Descent Results.
clf=RandomizedSearchCV(my_model,param_distributions=params,n_iter=5,scoring='f1_micro',n_jobs=-1,cv=5,verbose=3)

clf.fit(X_sm,y_sm)

print(clf.best_params_)

# Final model with hyperparameters
my_model = xgb.XGBClassifier(min_child_weight=3,max_depth=20,learning_rate=0.1,gamma=0.3,colsample_bytree=0.4)

my_model.fit(X_sm,y_sm)

ypred=my_model.predict(test)

pd.DataFrame(ypred).to_csv('xg_pred.csv')

