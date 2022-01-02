"""
## Part 7: Model Validation
"""
clf_RF = joblib.load("Models/clf_RF_hypertuned_selFeature.pkl")

clf_XBRF = joblib.load("Models/clf_XGRF_X_new.pkl")

# visualize confusion matrix

cm = confusion_matrix(y_test, y_RFpred, labels=clf_RF.classes_) # calculate value
disp = ConfusionMatrixDisplay(confusion_matrix=cm,              # display
                              display_labels=clf_RF.classes_)
disp.plot(); 
plt.show()


#----------------------------------------------------------------------------
#                       Uncertainty Random Forest                                                    
#----------------------------------------------------------------------------
## Compare uncertainty of Data and Model
scores = cross_val_score(clf_RF, X_new, y, cv=5, scoring='accuracy')
Udata = scores.std()

modAcuRF = []

for rs in range(1,6):
    model = RandomForestClassifier(random_state=random.randrange(rs))
    model.fit(X_train, y_train)
    modAcuRF += [accuracy_score(y_test, model.predict(X_test))]

Umodel = np.std(modAcuRF)

print("Uncertainty in the data: %.3f" % Udata)
print("Uncertainty in the model: %.3f" % Umodel)
print("The model performance is %.3f ± %.3f ± %.3f" % (scores.mean(),Udata,Umodel))

#----------------------------------------------------------------------------
#                       Uncertainty XGBOOST                                                  
#----------------------------------------------------------------------------

scores = cross_val_score(clf_XGRF, X_new, y, cv=5, scoring='accuracy')
Udata = scores.std()

params = dict(tree_method="exact", 
                eval_metric='mlogloss',
                use_label_encoder =False)

modAcuXGRF = []
for rs in range(1,6):
    model = xgboost.XGBClassifier(random_state=random.randrange(rs), **params)
    model.fit(X_train, y_train)
    modAcuXGRF += [accuracy_score(y_test, model.predict(X_test))]

Umodel = np.std(modAcuXGRF)

print("Uncertainty in the data: %.3f" % Udata)
print("Uncertainty in the model: %.3f" % Umodel)
print("The model performance is %.3f ± %.3f ± %.3f" % (scores.mean(),Udata,Umodel))