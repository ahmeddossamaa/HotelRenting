# import numpy as np
# import pandas as pd
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
from Preprocessing import preprocessing
# from sklearn.metrics import accuracy_score
from config.constants import TARGET_COLUMN, CURRENT_VERSION
from src.FeatureSelection import pearson
from src.Helpers import save, open_file,pickleStore
from src.Model import train_model,evaluate_model,RandomForestModel,MultipleLinearRegressor,SVRModel
from src.Helpers import getLatLng



def main():

    print("--------------------------------------- Splitting Phase Start ---------------------------------------")
    data = open_file("cls-nonulls-v1.csv")
    data = preprocessing(data)

   # dftr=open_file("dftr-v1.csv")
   # dfts=open_file("dfts-v1.csv")
   # dfv=open_file("dfv-v1.csv")

    '''
    print("--------------------------------------- Feature Selection Phase Start ---------------------------------------")

    f = pearson(dftr, 0.020)
    pickleStore(f, "features")
    dftr = dftr[f]
    dfts = dfts[f]
    # dfv = dfv[f]
    save(dftr, "dftr-f", CURRENT_VERSION)
    save(dfts, "dfts-f", CURRENT_VERSION)
    # save(dfv, "dfv-f", CURRENT_VERSION)

    print("--------------------------------------- Training Phase Start ---------------------------------------")
    # # Models for regression can be found in Model.py
    X_train= dftr.loc[:, dftr.columns != TARGET_COLUMN]
    y_train = dftr[TARGET_COLUMN]
    X_test = dfts.loc[:, dfts.columns != TARGET_COLUMN]
    y_test = dfts[TARGET_COLUMN]

    models_to_train = ['multiLinear', 'random_forest', 'svr']
    trained_models = {}
    for model_name in models_to_train:
        trained_models[model_name] = train_model(X_train, y_train, model_name)
    '''

    '''
    print("--------------------------------------- Testing Phase Start ---------------------------------------")
    for model_name, model in trained_models.items():
        mse, accuracy = evaluate_model(model, X_test, y_test)
        print(f"MSE {model_name}: {mse}")
        print(f"Accuracy {model_name}: {accuracy}")
    '''


    # You can use pickleStore and pickleOpen from Helpers.py to save your progress


# def main_old():
#     print("--------------------------------------- Training Phase Start ---------------------------------------")
#     data = open_file("Hotel_Renting_v4.csv")
#
#     X = data.loc[:, data.columns != TARGET_COLUMN]
#     y = data[TARGET_COLUMN]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#
#     df_training = pd.concat([X_train, y_train], axis=1)
#
#     df_training = preprocessing(df_training)
#
#     # df_training = open_file("columns-scaled-v1.csv")
#
#     #########################################################################################
#     print("--------------------------------------- Testing Phase Start ---------------------------------------")
#     f = pearson(df_training, 0.025)
#
#     save(df_training[f], "main-training", 1)
#
#     print(df_training[f])
#
#     df_testing = pd.concat([X_test, y_test], axis=1)
#
#     df_testing = preprocessing(df_testing, True)
#
#     X_test = df_testing.loc[:, df_testing.columns != TARGET_COLUMN]
#     y_test = df_testing[TARGET_COLUMN]
#
#     df_training = open_file("main-training-v1.csv")
#
#     X = df_training.loc[:, df_training.columns != TARGET_COLUMN]
#     y = df_training[TARGET_COLUMN]
#
#     model = treeModel(X, y)
#     prediction = model.predict(X_test, y_test)
#
#     mse = metrics.mean_squared_error(np.asarray(y_test), prediction)
#     # mse = np.mean(np.power((prediction - y_test), 2))
#
#     print(f"mse = {mse}")


main()