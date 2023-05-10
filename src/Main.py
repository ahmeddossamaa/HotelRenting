from Preprocessing import preprocessing, processNewColumns, GetMissingTripType, encodeColumns, scaleColumns, \
    fillTripType
from config.constants import TARGET_COLUMN, CURRENT_VERSION, ENCODE_COLS, REGRESSION_DATASET, CLASSIFICATION_DATASET
from src.FeatureSelection import pearson
from src.Helpers import save, open_file, pickleStore, split, pickleOpen
from src.Model import train_model, evaluate_model, RandomForestModel, MultipleLinearRegressor, SVRModel


def processing_v2(data, clf=False, isTesting=False):
    data = processNewColumns(data)

    # save(data, "processing-phase", 2)

    cols = ENCODE_COLS
    if clf:
        cols[TARGET_COLUMN] = {
            'label': True,
            'oneHot': False,
        },

    data = encodeColumns(data, cols=cols, isTesting=isTesting)

    # save(data, "encoding-phase", 1)

    data = fillTripType(data)

    # save(data, "fill-trip-type-phase", 1)

    data = scaleColumns(data)

    # save(data, "scaling-phase", 1)

    # save(data, "processed-columns-lr", CURRENT_VERSION)

    return data


def main_v2(file, clf=False):
    data = open_file(file)

    dftr, dfts = split(data)

    dftr = processing_v2(dftr, clf=clf, isTesting=False)
    dfts = processing_v2(dfts, clf=clf, isTesting=True)

    save(dftr, "dftr-reg" if not clf else "dftr-clf", CURRENT_VERSION)
    save(dfts, "dfts-reg" if not clf else "dfts-clf", CURRENT_VERSION)


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
    main_v2(REGRESSION_DATASET, clf=False)
    main_v2(CLASSIFICATION_DATASET, clf=True)
