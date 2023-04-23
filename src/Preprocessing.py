import re
import datetime as dt
import pandas as pd
from sklearn.model_selection import train_test_split
import concurrent.futures
from src.Model import TreeClassifier
from sklearn.metrics import accuracy_score
from config.constants import ADDRESS_COLUMN, TAGS_COLUMN, DATE_COLUMN
from src.Helpers import geoCoding, labelEncoding, featureScaling, extractNumberFromString, oneHotEncoding, save,  pickleStore, open_file, pickleOpen

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
    return open_file("columns-processing-v1.csv")
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
        data.drop(["Hotel_Address", "Negative_Review", "Positive_Review", "days_since_review", TAGS_COLUMN, DATE_COLUMN], axis=1, inplace=True)
        print("Redundant Columns Dropped!")

        save(data, "columns-processing", 1)
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
        if isTesting:
            encoders = pickleOpen("encoders")
        for i in cols:
            encoder = ''
            if cols[i]['label']:
                if encoders[i] is not None:
                    data[i] = labelEncoding(data[i], encoders[i])
                else:
                    encoder, data[i] = labelEncoding(data[i])
            elif cols[i]['oneHot']:
                if encoders[i] is not None:
                    x = oneHotEncoding(data[[i]], encoders[i])
                else:
                    encoder, x = oneHotEncoding(data[[i]])
                data.drop([i], axis=1, inplace=True)
                data = pd.concat([data, x], axis=1)

            if encoder != '':
                encoders[i] = encoder
        print("Columns Encoded!")

        save(data, "columns-encoded", 1)

        if not isTesting:
            print("Storing Encoders...")
            pickleStore(encoders, "encoders")
            print("Encoders Stored!")

        print("Scaling Columns...")
        for i in data:
            data[i] = featureScaling(data[i])
        print("Columns Scaled!")

        save(data, "columns-scaled", 1)
    except Exception as e:
        print("Error while processing in encodeAndScaleColumns:", e)

    print(data)

    return data


def preprocessing(data, isTesting=False):
    data.drop_duplicates()

    # def encode_column(column):
    #     if column not in data.keys():
    #         return pd.Series(dtype='float64')
    #     return featureScaling(labelEncoding(data[column]).astype('float64'))
    #
    # try:
    #     print("Encoding Columns...")
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         results = list(executor.map(encode_column, data.keys()))
    #     for i, column in enumerate(data.keys()):
    #         df[column] = results[i]
    #     print("Columns Encoded!")
    # except Exception as e:
    #     print("Error While Processing: ", e)

    return encodeAndScaleColumns(processNewColumns(data), isTesting)

def GetMissingTripType(df):
    #get the dataset
    
    #labelenconding 
    encoder, df.loc[:,'trv_type']  = labelEncoding(df.loc[:,'trv_type'])
    encoder, df.loc[:,'room_type'] = labelEncoding(df.loc[:,'room_type',])
    #df[['trv_type', 'room_type', 'days_number']] = featureScalingScikit(df[['trv_type', 'room_type', 'days_number']])
    dfOfNulls = df[df['trip_type'].isnull()]
    df = df.dropna()
    encoder, df.loc[:,'trip_type'] = labelEncoding(df.loc[:,'trip_type'])
    ##
    
    
    X = df[['trv_type', 'room_type', 'days_number']]
    
    y = df['trip_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    
    treecls = TreeClassifier(X_train, y_train)
    y_pred = treecls.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
     
    #there's nulls in days numbers!
    dfOfNulls = dfOfNulls[dfOfNulls['days_number'].isna() == False]
    dfOfNulls['trip_type'] = treecls.predict(dfOfNulls[['trv_type', 'room_type', 'days_number']])
    
    return dfOfNulls
