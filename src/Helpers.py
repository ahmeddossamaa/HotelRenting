from sklearn.preprocessing import LabelEncoder


# Encoding
def label_encoding(x, c):
    lbl = LabelEncoder()
    lbl.fit(list(x[c].values))
    x[c] = lbl.transform(list(x[c].values))
    return x


def one_hot_encoding(x):
    return x


# Scaling
def scaling(x):
    mini, maxi = min(x), max(x)

    if mini == maxi:
        return x

    x.iloc[:] = (x.iloc[:] - mini) / (maxi - mini)

    return x
