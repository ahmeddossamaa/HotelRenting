import re
import numpy as np
import pandas as pd
from flashgeotext.geotext import GeoText
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.Model import TreeClassifier
from sklearn.metrics import accuracy_score
from config.constants import ADDRESS_COLUMN, TAGS_COLUMN, DATE_COLUMN, CURRENT_VERSION, TARGET_COLUMN, ENCODE_COLS
from src.Helpers import geoCoding, featureScaling, extractNumberFromString, oneHotEncoding, save, \
    pickleStore, open_file, pickleOpen, encode, split, GetCity, GetCountry, initcitiesDict
from concurrent.futures import ThreadPoolExecutor
from flashgeotext.geotext import GeoText
from config.constants import ADDRESS_COLUMN, TAGS_COLUMN, DATE_COLUMN, CURRENT_VERSION, TARGET_COLUMN
from src.Helpers import geoCoding, labelEncoding, featureScaling, extractNumberFromString, oneHotEncoding, save, \
    pickleStore, open_file, pickleOpen, encode, one_hot_encode, split, initcitiesDict, GetCity, GetCountry, getLatLng
import concurrent.futures

"""def fix_date(x):
    for i in range(len(x)):
        if x[i].find("-") != -1:
            t = x[i].split("-")
        elif x[i].find("/") != -1:
            t = x[i].split("/")
        else:
            continue

        x[i] = float(dt.date(int(t[2]), int(t[1]), int(t[0])).timetuple().tm_yday)

    return pd.DataFrame({
        'Date': x
    }, dtype='float64')"""


def fix_date(a):
    date = pd.to_datetime(a, infer_datetime_format=True, dayfirst=True)
    return date.day, date.month, date.year


def process_tags_column(a):
    trip_type = ''
    trv_type = ''
    room_type = ''
    days_number = ''
    submitted = 0
    pet = 0

    try:
        b = eval(a)
        # remaining = []

        for i in b:
            i = i.lower().strip()

            check = lambda r: re.search(r, i)

            if check(r"submitted|mobile") and submitted == 0:
                submitted = 1
            elif check(r"pet") and pet == 0:
                pet = 1
            elif check(r"\btrip\b") and trip_type == '':
                trip_type = i
            elif check(r"night[s]?") and days_number == '':
                days_number = extractNumberFromString(i)
            elif check(
                    r"room[s]?|bed[s]?|suite[s]?|deluxe[s]?|standard|studio|apartment|king[s]?|queen[s]?") and room_type == '':
                room_type = i
            elif check(r"group|couple|solo|family|friend[s]?") and trv_type == '':
                trv_type = i
            # else:
            #     remaining.append(i)

    except:
        print(a)

    return trip_type, trv_type, room_type, days_number, submitted, pet


def processLongLat(a):
    res = geoCoding(a[ADDRESS_COLUMN])
    if res:
        res = res['features'][0]['geometry']['coordinates']
        print(f"res={res}")
        return res[1], res[0]
    return None, None


def processNewColumns(data):
    # return open_file("processed_trip_type-v1.csv")
    dateCols = ['review_day', 'review_month', 'review_year']
    tagsCols = ['trip_type', 'trv_type', 'room_type', 'days_number', 'submitted_by_mobile', 'with_pet']

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Tags
            print("Processing Tags...")
            tags_future = executor.submit(data[TAGS_COLUMN].apply(process_tags_column).apply, pd.Series)

            # Date
            print("Processing Date...")
            date_future = executor.submit(data[DATE_COLUMN].apply(fix_date).apply, pd.Series)

            # Days Since
            print("Processing Days Since Review...")
            days_since_future = executor.submit(data["days_since_review"].apply(extractNumberFromString).apply,
                                                pd.Series)

            # Getting City & Country
            print("Processing City & Country...")
            city_country_future = executor.submit(getCityAndCountry, data)

            # Wait for all futures to complete
            tags_result = tags_future.result()
            date_result = date_future.result()
            days_since_result = days_since_future.result()
            city_country_result = city_country_future.result()

            # Update data with processed columns
            data[tagsCols] = tags_result
            data[dateCols] = date_result
            data["days_since_review"] = days_since_result
            data = city_country_result

            # Drop Columns
            print("Dropping Redundant Columns...")
            data.drop(["Hotel_Address", "Negative_Review", "Positive_Review", "days_since_review", "Hotel_Address",
                       TAGS_COLUMN, DATE_COLUMN], axis=1, inplace=True)
            print("Redundant Columns Dropped!")

            # save(data, "processed-columns", CURRENT_VERSION)
    except Exception as e:
        print("Error while processing in processNewColumns:", e)

    return data


def encodeColumns(data, cols=ENCODE_COLS, isTesting=False, file="encoders"):
    # if isTesting:
    #     data = data.dropna()

    try:
        encoders = dict()
        print("Encoding Columns...")
        # print(data)

        if isTesting:
            encoders = pickleOpen(file)
        for i in cols:
            encoder = encoders[i] if i in encoders.keys() else None

            if cols[i]['label']:
                # data[i] = data[i].map(lambda d: d if d in encoder.classes_ else encoder.classes_)
                if isTesting:
                    data[i] = data[i].map(lambda d: d if d in encoder.classes_ else encoder.classes_)
                    pass
                else:
                    encoder = LabelEncoder()
                    encoder.fit(data[i])
                    encoder.classes_ = np.append(encoder.classes_, -1)

                data.loc[:, i] = encoder.transform(data[i])
            elif cols[i]['oneHot']:
                dummy_df = pd.get_dummies(data[i], prefix=i)

                data = pd.concat([data, dummy_df], axis=1)

                data = data.drop(i, axis=1)

            if encoder is not None:
                encoders[i] = encoder
        print("Columns Encoded!")

        # save(data, "encoded-columns", CURRENT_VERSION)

        if not isTesting:
            pickleStore(encoders, file)

        # save(data, "scaled-columns", CURRENT_VERSION)
    except Exception as e:
        print("Error while processing in encodeAndScaleColumns:", e)

    return data


def scaleColumns(data):
    try:
        print("Scaling Columns...")
        for i in data:
            data[i] = featureScaling(data[i])
        print("Columns Scaled!")
    except Exception as e:
        print("Error while processing in encodeAndScaleColumns:", e)

    return data


def preprocessing():
    data = open_file("processed-columns-v1.csv")

    data = data.drop_duplicates()
    data = data.dropna()

    data = data.reset_index(drop=True)

    if data['lng'].isna().sum() != 0:
        print("filling lat & lng nulls...")
        data = getLatLng(data)
        save(data, "cls-nonulls", CURRENT_VERSION)
        print("Done!")

    data = processNewColumns(data)
    data = encodeColumns(data, False)

    X = data.loc[:, data.columns != TARGET_COLUMN]
    y = data[TARGET_COLUMN]

    xtr, xts, ytr, yts = split(X, y)

    dftr = pd.concat([xtr, ytr], axis=1)
    dfts = pd.concat([xts, yts], axis=1)

    print(f"------------------------------------------------")
    print(f"len(dftr)={len(dftr)}")
    dftr = GetMissingTripType(dftr)
    print(f"len(dftr)={len(dftr)}")
    print(f"------------------------------------------------")

    print("--------------------------------------- Preprocessing Phase Start ---------------------------------------")
    dftr = encodeColumns(dftr, False)
    dfts = encodeColumns(dfts, True)

    # dftr = encodeAndScaleColumns(dftr, False)
    # dfts = encodeAndScaleColumns(dfts, True)
    # dfv = encodeAndScaleColumns(dfv, True)
    save(dftr, "dftr", CURRENT_VERSION)
    save(dfts, "dfts", CURRENT_VERSION)

    return dftr, dfts


def GetMissingTripType(df):
    dfts = df

    df = df.dropna()

    encoder = LabelEncoder()
    df.loc[:, 'trv_type'] = encode(df.loc[:, 'trv_type'], encoder)
    encoder.classes_ = np.append(encoder.classes_, -1)

    encoder2 = LabelEncoder()
    df.loc[:, 'room_type'] = encode(df.loc[:, 'room_type'], encoder2)
    encoder2.classes_ = np.append(encoder2.classes_, -1)

    encoder3 = LabelEncoder()
    df.loc[:, 'trip_type'] = encode(df.loc[:, 'trip_type'], encoder3)
    encoder3.classes_ = np.append(encoder3.classes_, -1)

    X = df[['trv_type', 'room_type', 'days_number']]

    y = df['trip_type']

    # dfts['trip_type'] = dfts['trip_type'].map(lambda d: d if d in encoder3.classes_ else -1)
    # dfts['trip_type'] = encoder3.transform(dfts['trip_type'])

    # df = df[df['trip_type'].isna()]
    dfts['trv_type'] = dfts['trv_type'].map(lambda d: d if d in encoder.classes_ else -1)
    dfts['trv_type'] = encoder.transform(dfts['trv_type'])

    dfts['room_type'] = dfts['room_type'].map(lambda d: d if d in encoder2.classes_ else -1)
    dfts['room_type'] = encoder2.transform(dfts['room_type'])

    treecls = TreeClassifier(X, y)
    for i in range(len(dfts)):
        # print(dfts.loc[i, :])
        if dfts.loc[i, 'trip_type'] == 0:
            # print(dfts.loc[i, 'trip_type'])
            # [dfts.loc[i, 'trv_type'], dfts.loc[i, 'room_type'], dfts.loc[i, 'days_number']]
            d = dfts.loc[i, ['trv_type', 'room_type', 'days_number']]
            print(d)
            yp = treecls.predict([d])
            # print(yp)
            dfts.loc[i, 'trip_type'] = encoder3.inverse_transform(yp)
    # dfts['trip_type'] = treecls.predict(df[['trv_type', 'room_type', 'days_number']])

    # dfts['trip_type'] = encoder3.inverse_transform(dfts['trip_type'])

    return dfts['trip_type']
    # # accuracy = accuracy_score(y_test, y_pred)
    # # print(accuracy)
    #
    # # there's nulls in days numbers!
    # dfOfNulls = dfOfNulls[dfOfNulls['days_number'].isna() == False]
    # dfOfNulls['trip_type'] = treecls.predict(dfOfNulls[['trv_type', 'room_type', 'days_number']])
    #
    # dfOfNulls = dfOfNulls.dropna()
    # data = pd.concat([df, dfOfNulls], axis=0)
    #
    # data['trv_type'] = encoder.inverse_transform(data['trv_type'])
    # data['room_type'] = encoder2.inverse_transform(data['room_type'])
    # data['trip_type'] = encoder3.inverse_transform(data['trip_type'])
    #
    # # save(data, "logistic-resultl", 1)
    #
    # return data


def getCityAndCountry(data):
    geotextCity = initcitiesDict()
    geotextCountry = GeoText()
    data.loc[:, 'Hotel_City'] = data.apply(lambda x: GetCity(x, geotextCity), axis=1)
    data.loc[:, 'Hotel_Country'] = data.apply(lambda x: GetCountry(x, geotextCountry), axis=1)
    return data


def testingPhasePreprocessing(data):
    data = data.drop_duplicates()

    data = data.dropna()

    data = processNewColumns(data)

    data = encodeColumns(data, isTesting=True)

    f = pickleOpen("features")
    print(data)
    data = data[f]

    return data


def fillTripType(dftr):
    dfts = pd.DataFrame(dftr)

    dftr.drop(dftr.loc[dftr['trip_type'] == 2].index, inplace=True)

    clf = TreeClassifier(dftr[['room_type', 'trv_type']], dftr['trip_type'])

    try:
        for i in range(len(dfts)):
            if dfts.iloc[i]['trip_type'] == 2:
                h = clf.predict([dfts.loc[i, ['room_type', 'trv_type']]])
                dfts.loc[i, 'trip_type'] = h[0]
    except NameError as e:
        print(e)

    return dfts
