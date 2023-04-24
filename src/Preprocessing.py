import re
import datetime as dt
import pandas as pd
from sklearn.model_selection import train_test_split
import concurrent.futures

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.Model import TreeClassifier
from sklearn.metrics import accuracy_score
from config.constants import ADDRESS_COLUMN, TAGS_COLUMN, DATE_COLUMN, CURRENT_VERSION
from src.Helpers import geoCoding, labelEncoding, featureScaling, extractNumberFromString, oneHotEncoding, save, \
    pickleStore, open_file, pickleOpen, encode

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
    b = eval(a)

    trip_type = ''
    trv_type = ''
    room_type = ''
    days_number = ''
    submitted = 0
    pet = 0
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
        # Tags
        print("Processing Tags...")
        data[tagsCols] = data[TAGS_COLUMN].apply(process_tags_column).apply(pd.Series)
        print("Tags Processed!")

        # Date
        print("Processing Date...")
        data[dateCols] = data[DATE_COLUMN].apply(fix_date).apply(pd.Series)
        print("Date Processed!")

        # Days Since
        print("Processing Days Since Review...")
        data["days_since_review"] = data["days_since_review"].apply(extractNumberFromString).apply(pd.Series)
        print("Days Since Review Processed!")

        # Drop Columns
        print("Dropping Redundant Columns...")
        data.drop(
            ["Hotel_Address", "Negative_Review", "Positive_Review", "days_since_review", TAGS_COLUMN, DATE_COLUMN],
            axis=1, inplace=True)
        print("Redundant Columns Dropped!")

        save(data, "processed-columns", CURRENT_VERSION)
    except Exception as e:
        print("Error while processing in processNewColumns:", e)

    return data


def encodeAndScaleColumns(data, isTesting):
    cols = {
        'trip_type': {
            'label': False,
            'oneHot': True,
        },
        'trv_type': {
            'label': True,
            'oneHot': False,
        },
        'room_type': {
            'label': True,
            'oneHot': False,
        },
        'Hotel_Name': {
            'label': True,
            'oneHot': False,
        },
        'Reviewer_Nationality': {
            'label': True,
            'oneHot': False,
        },
        'Hotel_Country': {
            'label': False,
            'oneHot': True,
        },
        'Hotel_City': {
            'label': False,
            'oneHot': True,
        },
    }

    try:
        encoders = dict()
        print("Encoding Columns...")
        # print(data)
        if isTesting:
            encoders = pickleOpen("encoders")
        for i in cols:
            encoder = encoders[i] if i in encoders.keys() else None
            # print(f"encoder={encoder}")

            if cols[i]['label']:
                if encoder is None:
                    # print(f"encoder={encoder}")
                    encoder = LabelEncoder()
                    print(f"encoder={encoder}")
                    # print(f"-----------------")

                data[i] = encode(data[i], encoder, testing=isTesting)
                # print(f"data[{i}]", data[i])
            elif cols[i]['oneHot']:
                if encoder is None:
                    encoder = OneHotEncoder()

                x = encode(data[[i]], encoder, testing=isTesting, label=False)
                # print(x)
                data = data.drop([i], axis=1)
                for j in x:
                    # print(j, "x = ", x[j])
                    data[j] = x[j]
                # print(data)
                # data = pd.concat([data, x], axis=1).drop_duplicates().reset_index(drop=True)
                # print(data)
                # print("---------------")

            if encoder is not None:
                encoders[i] = encoder
        print("Columns Encoded!")
        # print(data)
        save(data, "encoded-columns", CURRENT_VERSION)

        if not isTesting:
            print("Storing Encoders...")
            pickleStore(encoders, "encoders")
            print("Encoders Stored!")
        # return data
        print("Scaling Columns...")
        for i in data:
            data[i] = featureScaling(data[i])
        print("Columns Scaled!")

        save(data, "scaled-columns", CURRENT_VERSION)
    except Exception as e:
        print("Error while processing in encodeAndScaleColumns:", e)

    # print(data)

    return data


def preprocessing(data, isTesting=False):
    # data = processNewColumns(data)
    data = encodeAndScaleColumns(data, isTesting)

    return data


def GetMissingTripType(df):
    # get the dataset

    # labelenconding
    encoder, df.loc[:, 'trv_type'] = labelEncoding(df.loc[:, 'trv_type'])
    encoder, df.loc[:, 'room_type'] = labelEncoding(df.loc[:, 'room_type', ])
    # df[['trv_type', 'room_type', 'days_number']] = featureScalingScikit(df[['trv_type', 'room_type', 'days_number']])
    dfOfNulls = df[df['trip_type'].isnull()]
    df = df.dropna()
    # encoder, df.loc[:, 'trip_type'] = labelEncoding(df.loc[:, 'trip_type'])
    ##

    X = df[['trv_type', 'room_type', 'days_number']]

    y = df['trip_type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    treecls = TreeClassifier(X_train, y_train)
    y_pred = treecls.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    # there's nulls in days numbers!
    dfOfNulls = dfOfNulls[dfOfNulls['days_number'].isna() == False]
    dfOfNulls['trip_type'] = treecls.predict(dfOfNulls[['trv_type', 'room_type', 'days_number']])

    return dfOfNulls
