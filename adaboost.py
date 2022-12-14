import numpy as np
import pandas as pd
import gc
x=pd.read_csv("../input/train-mod/train_mod.csv")
y=pd.read_csv("../input/train-target/train_target.csv")
test=pd.read_csv("../input/test-mod/test_preprocessed.csv")
# from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=0)
X_sm,y_sm=rus.fit_resample(x,y)
del x
del y
gc.collect()
params={"n_estimators": [100,200, 300],
       "learning_rate": [0.05, 0.2, 1.0],
       "algorithm": ["SAMME","SAMME.R"],
       }
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

grid_search = RandomizedSearchCV(AdaBoostClassifier(), params, n_jobs = -1, scoring='f1_micro', verbose =1)
grid_result = grid_search.fit(X_sm,y_sm.values.ravel())
print(grid_result.best_params_)   

adb = AdaBoostClassifier(n_estimators = 300, learning_rate = 0.2, algorithm= 'SAMME.R')
np.mean(cross_val_score(adb,X_sm,y_sm,scoring="f1_micro", cv =2))
adb.fit(X_sm, y_sm);
del X_sm
del y_sm
gc.collect()
test=pd.read_csv("test_mod.csv")
ypred = adb.predict(test)
pd.DataFrame(ypred).to_csv('ada_pred2.csv')