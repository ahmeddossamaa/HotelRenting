from Preprocessing import preprocessing, processNewColumns, GetMissingTripType, encodeColumns, scaleColumns, \
    fillTripType
from config.constants import TARGET_COLUMN, CURRENT_VERSION, ENCODE_COLS, REGRESSION_DATASET
from src.FeatureSelection import pearson
from src.Helpers import save, open_file, pickleStore, split, pickleOpen
from src.Model import train_model, evaluate_model, RandomForestModel, MultipleLinearRegressor, SVRModel


def processing_v2(data, clf=False, isTesting=False):
    # data = data.drop_duplicates()

    # data = data.dropna()

    data = processNewColumns(data)

    # save(data, "processed-columns", 2)

    cols = ENCODE_COLS
    if clf:
        cols[TARGET_COLUMN] = {
            'label': True,
            'oneHot': False,
        },

    data = encodeColumns(data, cols=cols, isTesting=isTesting)

    data = fillTripType(data)

    # data = scaleColumns(data)

    save(data, "processed-columns-lr", CURRENT_VERSION)

    return data
    # if not isTesting:
    #     f = pearson(data, 0.02)
    #     pickleStore(f, 'features')
    # else:
    #     f = pickleOpen('features')
    #
    # data = data[f]
    #
    # return data

    # save(data, "processed-columns-lr", CURRENT_VERSION)

    # print(data[data['days_number'].isna()])


def main():
    print("--------------------------------------- Splitting Phase Start ---------------------------------------")
    # dftr, dfts = preprocessing()
    dftr = open_file("dftr-v1.csv")
    dfts = open_file("dfts-v1.csv")

    print(
        "--------------------------------------- Feature Selection Phase Start ---------------------------------------")
    f = pearson(dftr, 0.020)
    pickleStore(f, "features")
    dftr = dftr[f]
    dfts = dfts[f]

    save(dftr, "dftr-f", CURRENT_VERSION)
    save(dfts, "dfts-f", CURRENT_VERSION)

    print("--------------------------------------- Training Phase Start ---------------------------------------")
    # # Models for regression can be found in Model.py
    X_train = dftr.loc[:, dftr.columns != TARGET_COLUMN]
    y_train = dftr[TARGET_COLUMN]
    X_test = dfts.loc[:, dfts.columns != TARGET_COLUMN]
    y_test = dfts[TARGET_COLUMN]

    models_to_train = ['multiLinear', 'random_forest', 'svr']
    trained_models = {}
    for model_name in models_to_train:
        trained_models[model_name] = train_model(X_train, y_train, model_name)
    print("--------------------------------------- Testing Phase Start ---------------------------------------")
    for model_name, model in trained_models.items():
        mse, accuracy = evaluate_model(model, X_test, y_test)
        print(f"MSE {model_name}: {mse}")
        print(f"Accuracy {model_name}: {accuracy}")


if __name__ == '__main__':
    processing_v2(open_file(REGRESSION_DATASET))
