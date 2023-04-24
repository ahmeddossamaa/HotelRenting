import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from Preprocessing import process_tags_column, processLongLat, preprocessing, processNewColumns, GetMissingTripType
from config.constants import TARGET_COLUMN, CURRENT_VERSION
from src.FeatureSelection import pearson
from src.Helpers import save, open_file, split
from src.Model import treeModel


def main_v2():
    print("--------------------------------------- Splitting Phase Start ---------------------------------------")
    # data = open_file("Hotel_Renting_v4.csv")
    #
    # data = data.drop_duplicates()
    #
    # data = processNewColumns(data)

    data = open_file("processed-columns-v1.csv")

    X = data.loc[:, data.columns != TARGET_COLUMN]
    y = data[TARGET_COLUMN]

    xtr, xts, xv, ytr, yts, yv = split(X, y)

    # data = open_file("hotel-dataset-processed-v1.csv")

    # xtr = data.loc[:, data.columns != TARGET_COLUMN]
    # ytr = data[TARGET_COLUMN]
    # print(data)
    # print(xtr)
    # print(xts)

    dftr = pd.concat([xtr, ytr], axis=1)
    dfts = pd.concat([xts, yts], axis=1)
    dfv = pd.concat([xv, yv], axis=1)

    dftr = GetMissingTripType(dftr)

    print("--------------------------------------- Preprocessing Phase Start ---------------------------------------")
    dftr = preprocessing(dftr)
    dfts = preprocessing(dfts, True)
    dfv = preprocessing(dfv, True)

    save(dftr, "dftr", CURRENT_VERSION)
    save(dfts, "dfts", CURRENT_VERSION)
    save(dfv, "dfv", CURRENT_VERSION)

    print("--------------------------------------- Feature Selection Phase Start ---------------------------------------")
    f = pearson(dftr, 0.025)

    dftr = dftr[f]
    dfts = dfts[f]
    dfv = dfv[f]

    save(dftr, "dftr-f", CURRENT_VERSION)
    save(dfts, "dfts-f", CURRENT_VERSION)
    save(dfv, "dfv-f", CURRENT_VERSION)

    print("--------------------------------------- Training Phase Start ---------------------------------------")
    # Models for regression can be found in Model.py
    print("--------------------------------------- Testing Phase Start ---------------------------------------")
    # You can use pickleStore and pickleOpen from Helpers.py to save your progress


def main():
    print("--------------------------------------- Training Phase Start ---------------------------------------")
    data = open_file("Hotel_Renting_v4.csv")

    X = data.loc[:, data.columns != TARGET_COLUMN]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    df_training = pd.concat([X_train, y_train], axis=1)

    df_training = preprocessing(df_training)

    # df_training = open_file("columns-scaled-v1.csv")

    #########################################################################################
    print("--------------------------------------- Testing Phase Start ---------------------------------------")
    f = pearson(df_training, 0.025)

    save(df_training[f], "main-training", 1)

    print(df_training[f])

    df_testing = pd.concat([X_test, y_test], axis=1)

    df_testing = preprocessing(df_testing, True)

    X_test = df_testing.loc[:, df_testing.columns != TARGET_COLUMN]
    y_test = df_testing[TARGET_COLUMN]

    df_training = open_file("main-training-v1.csv")

    X = df_training.loc[:, df_training.columns != TARGET_COLUMN]
    y = df_training[TARGET_COLUMN]

    model = treeModel(X, y)
    prediction = model.predict(X_test, y_test)

    mse = metrics.mean_squared_error(np.asarray(y_test), prediction)
    # mse = np.mean(np.power((prediction - y_test), 2))

    print(f"mse = {mse}")


if __name__ == "__main__":
    main_v2()
