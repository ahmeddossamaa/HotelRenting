import re
import datetime as dt
import pandas as pd
import concurrent.futures
from config.constants import ADDRESS_COLUMN, TAGS_COLUMN, DATE_COLUMN
from src.Helpers import geoCoding, labelEncoding, featureScaling, extractNumberFromString


def fix_date(x):
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
    }, dtype='float64')


def fix_date_v2(a):
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
    # if a['lat'] is not None:
    #     return a
    # print(f"a={a}")
    # return None, None
    res = geoCoding(a[ADDRESS_COLUMN])
    if res:
        res = res['features'][0]['geometry']['coordinates']
        print(f"res={res}")
        return res[1], res[0]
    return None, None


def preprocessing(data):
    dateCols = ['review_day', 'review_month', 'review_year']
    tagsCols = ['trip_type', 'trv_type', 'room_type', 'days_number', 'submitted_by_mobile', 'with_pet']

    df = pd.DataFrame()

    data.drop(["Hotel_Address", "lat", "lng",
               "Negative_Review", "Review_Total_Negative_Word_Counts",
               "Total_Number_of_Reviews", "Positive_Review",
               "Review_Total_Positive_Word_Counts",
               "Total_Number_of_Reviews_Reviewer_Has_Given"
               ], axis=1, inplace=True)

    try:
        # Tags
        print("Processing Tags...")
        data[tagsCols] = data[TAGS_COLUMN].apply(process_tags_column).apply(pd.Series)
        data.drop([TAGS_COLUMN], axis=1, inplace=True)
        print("Tags Processed!")

        # Date
        print("Processing Date...")
        data[dateCols] = data[DATE_COLUMN].apply(fix_date_v2).apply(pd.Series)
        data.drop([DATE_COLUMN], axis=1, inplace=True)
        print("Date Processed!")

        # Days Since
        print("Processing Days Since Review...")
        data["days_since_review"] = data["days_since_review"].apply(extractNumberFromString).apply(pd.Series)
        data.drop(["days_since_review"], axis=1, inplace=True)
        print("Days Since Review Processed!")
    except Exception as e:
        print("Error While Processing: ", e)

    def encode_column(column):
        if column not in data.keys():
            return pd.Series(dtype='float64')
        return featureScaling(labelEncoding(data[column]).astype('float64'))

    try:
        print("Encoding Columns...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(encode_column, data.keys()))
        for i, column in enumerate(data.keys()):
            df[column] = results[i]
        print("Columns Encoded!")
    except Exception as e:
        print("Error While Processing: ", e)

    return df
