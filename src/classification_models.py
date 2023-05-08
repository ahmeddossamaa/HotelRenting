# Import LogisticRegression() model from scikit_learn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def getAccuracy(y, h):
    return accuracy_score(y, h)


# multi class classification
def multi_class(x, y, ovr='ovr'):
    clf = LogisticRegression(multi_class=ovr)
    clf.fit(x, y)
    return clf


def SVMKernelsClf(x, y, kernel='rbf'):
    clf = SVC(kernel=kernel)
    clf.fit(x, y)
    return clf


def RandomForestClf(x, y, n=100, r=40):
    clf = RandomForestClassifier(n_estimators=n, random_state=r)
    clf.fit(x, y)
    return clf


"""
# Plots the Probability Distributions and the ROC Curves One vs Rest
plt.figure(figsize=(12, 8))
bins = [i / 20 for i in range(20)] + [1]
classes = Multiclass_model.classes_
roc_auc_ovr = {}
for i in range(len(classes)):
    # Gets the class
    c = classes[i]

    # Prepares an auxiliar dataframe to help with the plots
    df_aux = X_train.copy()
    df_aux['class'] = [1 if y == c else 0 for y in y_train]
    df_aux['prob'] = y_proba[:, i]
    df_aux = df_aux.reset_index(drop=True)

    # Plots the probability distribution for the class and the rest
    ax = plt.subplot(2, 3, i + 1)
    sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
    ax.set_title(c)
    ax.legend([f"Class: {c}", "Rest"])
    ax.set_xlabel(f"P(x = {c})")

    # Calculates the ROC Coordinates and plots the ROC Curves
    ax_bottom = plt.subplot(2, 3, i + 4)
    tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
    plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
    ax_bottom.set_title("ROC Curve OvR")

    # Calculates the ROC AUC OvR
    roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])
plt.tight_layout()
"""
