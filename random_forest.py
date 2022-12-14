import numpy as np
import pandas as pd
import gc
x=pd.read_csv("train_mod.csv")
y=pd.read_csv("target_mod.csv")
# from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=0)
X_sm,y_sm=rus.fit_resample(x,y)
del x
del y
gc.collect()
params={"n_estimators": [100,200,300],
       "min_samples_split" : [2],
       "max_features" : ["sqrt","log",30],
       "n_jobs" : [-1],
       "max_samples" : [0.1,0.5,1.0],
       'verbose' : [1]
       }
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

grid_search = GridSearchCV(estimator = RandomForestClassifier(),param_grid= params ,  n_jobs = 1, cv =5, scoring='f1_micro')
grid_result = grid_search.fit(X_sm,y_sm.values.ravel())
print(grid_result.best_params_)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rfc = RandomForestClassifier(min_samples_split=2,max_features='sqrt',n_estimators=200, verbose = 1, n_jobs=-1,max_samples=1.0)
np.mean(cross_val_score(rfc,X_sm,y_sm,scoring="f1_micro", cv =5))
rfc.fit(X_sm, y_sm)
del X_sm
del y_sm
gc.collect()
test=pd.read_csv("test_mod.csv")
ypred = rfc.predict(test)
pd.DataFrame(ypred).to_csv("random_pred5.csv")
