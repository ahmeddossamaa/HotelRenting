from sklearn.feature_selection import SelectKBest, f_classif


def anova(x, y, k):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(x, y)

    return selector.get_support(indices=True)


def pearson(d, t, l):
    corr = d.corr()

    return corr.index[abs(corr[t]) > l].delete(t)
