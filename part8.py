## Visualise the model!

# Extract single tree
estimator = clf_RF.estimators_[10]
classes = list(set(zip(y_, y)))
classes.sort(key=lambda y: y[1])
classes_sorted = list(zip(*classes))[0]

from sklearn.tree import export_graphviz

from part4_featureSelection import X_new
# Export as dot file
export_graphviz(estimator, 
                out_file='tree.dot', 
                feature_names = None,
                class_names = classes_sorted,
                rounded = True, proportion = False, 
                precision = 2, filled = True)


# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])