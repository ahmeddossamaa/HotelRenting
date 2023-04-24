from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.Helpers import labelEncoding, open_file
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


def treeModel(X, y):
    model = tree.DecisionTreeRegressor()
    return model.fit(X, y)


def logisticModel():
    df = open_file("columns-processing-v1.csv")
    # df = pd.read_csv('columns-processing-v1.csv')
    df = df.dropna()

    encoder, df['trv_type'] = labelEncoding(df['trv_type'])
    encoder, df['room_type'] = labelEncoding(df['room_type'])
    encoder, df['trip_type'] = labelEncoding(df['trip_type'])

    X = df[['trv_type', 'room_type', 'days_number']]
    y = df['trip_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    clf = DecisionTreeClassifier(random_state=40)
    clf.fit(X_train, y_train)
    # y_pred = clf.predict([[df['trv_type'][14],df['room_type'][14],df['days_number'][14]]])
    y_pred = clf.predict(X_test)
    for i, j in enumerate(encoder.inverse_transform(y_pred)):
        print(i, j)

    # print()
    # print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)


# regression models (X -> X_train, y -> y_train)
def LinearModel(X, y):
    lr = LinearRegression()
    lr.fit(X, y)
    return lr


'''use more than one tree to make a predict'''


def RandomForestModel(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X, y)
    return rf  # to use the object in predicting valid & testing dataset (should we save it?)


'''create a n-1 D plane'''


def SVRModel(X, y):
    svr = SVR(kernel='rbf', gamma=0.1)  # kernal = linear, poly, sgmoid (try all and choice the higher acc)
    svr.fit(X, y)
    return svr


# ++  gradient boosting regression
##

# classifier for trip type
def TreeClassifier(X, Y):
    clf = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_split=15)
    clf.fit(X, Y)
    return clf
