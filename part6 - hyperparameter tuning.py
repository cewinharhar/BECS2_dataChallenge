"""
## Part 6: Hyperparameter Tuning
In this chapter the model is optimized by hyperparameter tuning. A random grid search is applied to selected hyperparameters of both models. The hyperparametertuning follows the instructions of the following publication https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 (31.12.21).
"""

X_train, X_test, y_train, y_test = joblib.load("Models/X_y_split.pkl")

#Hyperparameter tuning for the random forest classifier using random grid search
#get model parameters(delete after hyperparameter tuning)
#params = clf_RF.get_params()
#params

#----------------------------------------------------------------------------
                            #Jonathan                            
#----------------------------------------------------------------------------
#Define hyperparameters for tuning
n_estimators_RF = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] #number of trees
max_features_RF = ['auot', 'sqrt'] #number of features
max_depth_RF = [int(x) for x in np.linspace(10, 110, num=1)] #numbber of levels
max_depth_RF.append(None)
min_samples_split_RF = [2, 5, 10] #minimum number of samples required to split a note
min_samples_leaf_RF = [1, 2, 4] #minimum number of samples required at each leaf node
bootstrap_RF = [True, False] #method of selecting samples for training

#initialize random grid
random_grid = {'n_estimators': n_estimators_RF,
            'max_features': max_features_RF,
            'max_depth': max_depth_RF,
            'min_samples_split': min_samples_split_RF,
            'min_samples_leaf': min_samples_leaf_RF,
            'bootstrap': bootstrap_RF}

#define model parameters for random grid search
RF_random = RandomizedSearchCV(estimator = clf_RF,
            param_distributions=random_grid, n_iter=200,
            cv=5, verbose=2, random_state=13, n_jobs=-1)

#fit the random search model
RF_random.fit(X_train, y_train)

#get best hyperparameters from the model
RF_random.best_params_

#run model with optimized hyperparameters
clf_RF_tuned = RandomForestClassifier(n_estimators=400,
                                    min_samples_split=2,
                                    min_samples_leaf=4,
                                    max_features='sqrt',
                                    max_depth= None,
                                    bootstrap= True,
                                    random_state=1)
clf_RF_tuned.fit(X_train ,y_train)
y_RFpred_tuned = clf_RF_tuned.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy Random Forest:",metrics.accuracy_score(y_test , y_RFpred_tuned))

params = clf_RF.get_params()
params

#Hyperparameter tuning for the random forest classifier using random grid search
#get model parameters(delete after hyperparameter tuning)
params = clf_XGRF.get_params()
params


#----------------------------------------------------------------------------
                            #Kevin                        
#----------------------------------------------------------------------------

clf_RF = RandomForestClassifier()
# second try hyperparameter tuning

#Hyperparameter tuning for the random forest classifier using random grid search
#get model parameters(delete after hyperparameter tuning)
#params = clf_RF.get_params()
#params\n,

#Define hyperparameters for tuning\n,
n_estimators_RF = [x for x in np.linspace(start=50, stop=500, num=15, dtype=int)] #number of trees
criterion_RF    = ["gini", "entropy"]

max_depth_RF = [int(x) for x in np.arange(1, 20)] #numbber of levels
max_depth_RF.append(None)


min_samples_split_RF = [1, 2, 5, 10] #minimum number of samples required to split a note
min_samples_leaf_RF = [np.arange(start=1, stop=5)] #minimum number of samples required at each leaf node

max_features_RF = ['auto', 'sqrt', "log2"] #number of features

class_weight        = []  #<---     SUPER WICHTIG FÜR BIOMARKER

bootstrap_RF = [True, False] #method of selecting samples for training
#initialize random grid \n,
# RANDOM FOREST PARAMS\n,
random_grid = {'n_estimators'   : n_estimators_RF,
            'max_features'      : max_features_RF,
            'max_depth'         : max_depth_RF,                        
            'min_samples_split' : min_samples_split_RF,
            'bootstrap'         : bootstrap_RF}

#define model parameters for random grid search
RF_random = RandomizedSearchCV(estimator = clf_RF,
            param_distributions=random_grid, n_iter=200,
            cv=7, verbose=0, n_jobs=-1)

#fit the random search model
RF_random.fit(X_train, y_train)

#get best hyperparameters from the model
RF_random.best_params_


params = {'n_estimators': 114, 'min_samples_split': 2, 'max_features': 'auto', 'max_depth': 18, 'bootstrap': False} # acc: 0.6363

clf_RF = RandomForestClassifier(**params)
clf_RF.fit(X_train, y_train) 

y_RFpred = clf_RF.predict(X_test)

# Model Accuracy, how often is the classifier correct?\n,
print("Accuracy Random Forest: ",metrics.accuracy_score(y_test , y_RFpred))
print(classification_report(y_test, y_RFpred))

#--------------------------  XGBOOST  ------------------------------------

