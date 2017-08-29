from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

clf_decTree = tree.DecisionTreeClassifier()
clf_KNN = KNeighborsClassifier()

X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],[171,75,42],[181,85,43]]

Y = ['m','m','f','f','m','m','f','f','f','m','m']


clf_decTree = clf_decTree.fit(X, Y)
clf_KNN.fit(X, Y)

pred_tree = clf_decTree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Precisao DecisionTree: {}'.format(acc_tree))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Precisao KNN: {}'.format(acc_KNN))
