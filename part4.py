"""
### Part 4: Creation of Data Sets, Feature and Model Selection
* Create pipeline for imputing, scaling !! **Scaling is not needed for Random Forest**
* (https://towardsdatascience.com/how-data-normalization-affects-your-random-forest-algorithm-fbc6753b4ddf)
* Creation of training, validation and test sets
* Feature Selection, Engineering
* Model Selection

"""

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

"""
## Feature selection

### Two possibilities:
* Tree-based feature selection - sklearn.feature_selection.SelectFromModel
    * May be used with sklearn.tree models
##
* Sequential Feature Selection - sklearn.feature_selection.SequentialFeatureSelector
    * May be used with xgboost
"""

# Tree Feature Selection 

"""
Can be used in pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
"""
#create Random Forest classifier with default hyperparameters
raFo = RandomForestClassifier(random_state=1)
raFo = raFo.fit(X, y)

#checkout importance in a histogram
plt.hist(raFo.feature_importances_, bins=100)
plt.title("Histogram of the feature importance for all 2730 proteins")
plt.xlabel("Importance")
plt.ylabel("Count")


#get the reduced X
model = SelectFromModel(estimator = raFo, prefit=True)
X_new = model.transform(X)

print(f"Original X shape: {X.shape}")
print(f"Feature selected X_new shape: {X_new.shape}")


# sequential Feature Selection 

params = dict(tree_method="exact", 
                eval_metric='mlogloss',
                use_label_encoder =False)

clf_XGRF = xgboost.XGBClassifier(random_state=7, **params)
model = SequentialFeatureSelector(estimator = clf_XGRF, n_features_to_select = 0.20, cv = 10,  n_jobs=-1)
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
