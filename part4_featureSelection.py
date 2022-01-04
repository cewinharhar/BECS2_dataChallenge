"""
### Part 4: Creation of Data Sets, Feature and Model Selection
* Create pipeline for imputing, scaling !! **Scaling is not needed for Random Forest**
* (https://towardsdatascience.com/how-data-normalization-affects-your-random-forest-algorithm-fbc6753b4ddf)
* Creation of training, validation and test sets
* Feature Selection, Engineering
* Model Selection

"""
import os
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from sklearn_genetic import GAFeatureSelectionCV
import part1
#-----------------------  Inputing  / Encoder  -----------------------------------

#Make pipeline
dataPrepPipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ]) 

#X is already purely numerical
X = dataPrepPipe.fit_transform(X_)

# Encode target labels
labEnc = preprocessing.LabelEncoder() 
y = labEnc.fit_transform(y_) 
set(zip(y_, y))

# Save X and y for next session
joblib.dump(y, "Models/y.pkl")
joblib.dump(X, "Models/X.pkl")

# load saved X and y 
y = joblib.load("Models/y.pkl")
X = joblib.load("Models/X.pkl")



#----------------------------------------------------------------------------
#                       Feature Selection                          
#----------------------------------------------------------------------------
"""
## Feature selection

### Two possibilities:
* Tree-based feature selection - sklearn.feature_selection.SelectFromModel
    * May be used with sklearn.tree models
##
* Sequential Feature Selection - sklearn.feature_selection.SequentialFeatureSelector
    * May be used with xgboost
"""


#----------------------------------------------------------------------------
#                        Random Forest: Tree Feature Selection                          

"""
Can be used in pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
"""
#create Random Forest classifier with default hyperparameters
params  = {'n_estimators': 114, 'min_samples_split': 2, 'max_features': 'auto', 'max_depth': 18, 'bootstrap': False}
raFo = RandomForestClassifier(random_state=0, **params)
raFo = raFo.fit(X, y)

#checkout importance in a histogram
plt.hist(raFo.feature_importances_, bins=100)
plt.title("Histogram of the feature importance for all 2730 proteins")
plt.xlabel("Importance")
plt.ylabel("Count")


#get the reduced X
model = SelectFromModel(estimator = raFo, prefit=True,)
X_new = model.transform(X)

print(f"Original X shape: {X.shape}")
print(f"Feature selected X_new shape: {X_new.shape}")

joblib.dump(X_new, "Models/X_new.pkl")

#----------------------------------------------------------------------------
#            Random Forest: Evolutionary Algorythm Feature selection

params  = {'n_estimators': 114, 'min_samples_split': 2, 'max_features': 'auto', 'max_depth': 18, 'bootstrap': False}
clf_RF    = RandomForestClassifier(random_state=0, **params)

evolved_estimator = GAFeatureSelectionCV(
    estimator   = clf_RF,
    cv          = None,
    population_size=20, 
    generations =40,
    crossover_probability=0.8,
    mutation_probability = 0.15,
    n_jobs      = -1,
    scoring     = "accuracy")

# Train and select the features
evolved_estimator.fit(X, y)

# Features selected by the algorithm
features= evolved_estimator.best_features_

X_RFGA    = X[:, features]

joblib.dump(X_RFGA, "Models/X_RFGA.pkl")
joblib.dump(features, "Models/featSel_RFGA.pkl")

#----------------------------------------------------------------------------
#                   XGboost:  Feature Selection                          

params = dict(tree_method="exact", 
                eval_metric='mlogloss',
                use_label_encoder =False)

clf_XGRF = xgboost.XGBClassifier(random_state=0, **params)
clf_XGRF.fit(X,y)

tresh = np.linspace(start=0.05, stop=0.3, num=10)
for tre in tresh:

    model = SelectFromModel(estimator = clf_XGRF, n_jobs=-1, treshhold = tre)
    model.fit(X,y)

X_new = model.transform(X)

#checkout importance in a histogram
plt.hist(raFo.feature_importances_, bins=100)

print(f"Original X shape: {X.shape}")
print(f"Feature selected X_new shape: {X_new.shape}")






# Visualize feature importance

importances = raFo.feature_importances_

std = np.std([tree.feature_importances_ for tree in raFo.estimators_], axis=0)
std.sort()

forest_importances = pd.Series(importances)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
ax.get_xaxis().set_visible(False)
fig.tight_layout()
plt.show()

                                               
