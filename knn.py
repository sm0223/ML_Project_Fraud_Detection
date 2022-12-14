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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#grid_search
params={"weights": ["uniform", "distance"],
       "n_neighbors": [4, 6, 8, 10],
        "algorithm ":['auto']
       }
grid_search = RandomizedSearchCV(KNeighborsClassifier(), params, scoring = 'f1_micro', n_jobs = -1)
grid_result = grid_search.fit(X_sm,y_sm)
#cross validation
knn = KNeighborsClassifier(weights = "distance", n_neighbors = 6, algorithm = 'auto')
np.mean(cross_val_score(knn,X_sm,y_sm,scoring="f1_micro", cv =3))
# fitting the model and getting prediction
ypred = knn.fit(X_sm, y_sm)
ypred = knn.predict(test)
pd.DataFrame(ypred).to_csv("knn_ht_pred.csv")