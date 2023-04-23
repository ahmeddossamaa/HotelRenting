from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


#regression models (X -> X_train, y -> y_train)
def LinearRegression(X,y):
    lr  = LinearRegression()
    lr.fit(X, y)
    return lr

'''use more than one tree to make a predict'''
def RandomForestRegressor(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X, y)
    return rf  #to use the object in predicting valid & testing dataset (should we save it?)

'''create a n-1 D plane'''
def SVR(X,y):
    svr = SVR(kernel='rbf', gamma=0.1) #kernal = linear, poly, sgmoid (try all and choice the higher acc)
    svr.fit(X, y)
    return svr

#++  gradient boosting regression
##

#classifier for trip type
def TreeClassifier(X,Y):
    clf = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_split=15)
    clf.fit(X, Y)
    return clf

 
