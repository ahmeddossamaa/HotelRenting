import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from Preprocessing import process_tags_column, processLongLat, preprocessing
from config.constants import TARGET_COLUMN
from src.FeatureSelection import pearson
from src.Helpers import save, open_file
from src.Model import treeRegression

data = open_file("Hotel_Renting_v4.csv")


def main():
    X = data.loc[:, data.columns != TARGET_COLUMN]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # df_training = pd.concat([X_train, y_train], axis=1)

    # df_training = preprocessing(df_training)

    # df_training = open_file("columns-scaled-v1.csv")

    #########################################################################################

    # f = pearson(df_training, 0.025)

    # save(df_training[f], "main-training", 1)

    # print(df_training[f])

    df_testing = pd.concat([X_test, y_test], axis=1)

    df_testing = preprocessing(df_testing, True)

    X_test = df_testing.loc[:, df_testing.columns != TARGET_COLUMN]
    y_test = df_testing[TARGET_COLUMN]

    df_training = open_file("main-training-v1.csv")

    X = df_training.loc[:, df_training.columns != TARGET_COLUMN]
    y = df_training[TARGET_COLUMN]

    model = treeRegression(X, y)
    prediction = model.predict(X_test, y_test)

    mse = metrics.mean_squared_error(np.asarray(y_test), prediction)
    # mse = np.mean(np.power((prediction - y_test), 2))

    print(f"mse = {mse}")


if __name__ == "__main__":
    main()
