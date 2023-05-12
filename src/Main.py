import concurrent.futures

from Preprocessing import preprocessing, processNewColumns, GetMissingTripType, encodeColumns, scaleColumns, \
    fillTripType
from config.constants import TARGET_COLUMN, CURRENT_VERSION, ENCODE_COLS, REGRESSION_DATASET, CLASSIFICATION_DATASET, \
    PRODUCTION
from src.FeatureSelection import pearson
from src.Helpers import save, open_file, pickleStore, split, pickleOpen
from src.Model import train_model, evaluate_model, RandomForestModel, MultipleLinearRegressor, SVRModel


def processing_v2(data, output, v=CURRENT_VERSION, clf=False, isTesting=False):
    file = "encoders-clf" if clf else "encoders-reg"

    if PRODUCTION:
        data = processNewColumns(data)

    # v = 5 if not clf else 6

    # save(data, 'processing-phase', v)

    # data = open_file(f"processing-phase-v{v}.csv")

    cols = ENCODE_COLS
    if clf:
        cols[TARGET_COLUMN] = dict({
            'label': True,
            'oneHot': False,
        })

    data = encodeColumns(data, cols=cols, isTesting=isTesting, file=file)

    save(data, "encoding-phase", v)

    data = fillTripType(data, file=file)

    save(data, "filling-phase", v)

    data = scaleColumns(data)

    # save(data, "scaling-phase", v)

    save(data, output, v)

    return data


def main_v2(file, clf=False):
    data = open_file(file)

    if not PRODUCTION:
        print(f"processNewColumns {clf}...")
        data = processNewColumns(data)
        # save(data, "processNewColumns", CURRENT_VERSION)

    dftr, dfts = split(data)

    dftr = processing_v2(dftr, "dftr-reg" if not clf else "dftr-clf", v=CURRENT_VERSION, clf=clf, isTesting=False)
    dfts = processing_v2(dfts, "dfts-reg" if not clf else "dfts-clf", v=CURRENT_VERSION, clf=clf, isTesting=True)


def main():
    print("--------------------------------------- Splitting Phase Start ---------------------------------------")
    # dftr, dfts = preprocessing()
    dftr = open_file("dftr-v1.csv")
    dfts = open_file("dfts-v1.csv")

    print("--------------------------------------- Feature Selection Phase Start ---------------------------------------")
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
    # with concurrent.futures.ThreadPoolExecutor() as e:
    #     reg = e.submit(main_v2, REGRESSION_DATASET, clf=False)
    #     clf = e.submit(main_v2, CLASSIFICATION_DATASET, clf=True)
    # reg.result()
    # clf.result()
    # main_v2(REGRESSION_DATASET, clf=False)
    main_v2(CLASSIFICATION_DATASET, clf=True)
    # processing_v2(open_file(CLASSIFICATION_DATASET), "clf-data", clf=True)
    # data = open_file(CLASSIFICATION_DATASET)

    # print(data.isna().sum())
