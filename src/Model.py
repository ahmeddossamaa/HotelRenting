import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.Helpers import labelEncoding, open_file


def treeRegression(X, y):
    model = tree.DecisionTreeRegressor()
    return model.fit(X, y)


def logisticRegression():
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
