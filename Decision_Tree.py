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

pd.set_option('display.max_columns',None)

x=pd.read_csv("train_mod.csv")
y=pd.read_csv("train_target.csv")
test=pd.read_csv("test_preprocessed.csv")

from sklearn.model_selection import GridSearchCV,KFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

rus=RandomUnderSampler(random_state=0)
X_sm,y_sm=rus.fit_resample(x,y)

params={"max_depth": [2,3,5,20,50],
       "min_samples_leaf": [5,10,20,50,100],
       "criterion": ["gini","entropy"]
       }

from sklearn import tree
dt = tree.DecisionTreeClassifier()
clf=GridSearchCV(estimator=dt,param_grid=params,scoring='f1_micro')

clf.fit(X_sm,y_sm)

print(clf.best_estimator_)

print(clf.best_score_)

dt=tree.DecisionTreeClassifier(criterion='entropy', max_depth=50, min_samples_leaf=50)

dt.fit(X_sm,y_sm)

ypred=dt.predict(test)

pd.DataFrame(ypred).to_csv('dt_pred.csv')


##Kaggle score = 0.74
