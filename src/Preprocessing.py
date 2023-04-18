import re
import datetime as dt
import pandas as pd
import concurrent.futures
from config.constants import ADDRESS_COLUMN, TAGS_COLUMN
from src.Helpers import geoCoding, labelEncoding, featureScaling


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
            res = re.search('[0-9]+', i)
            days_number = res.group() if res else ''
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
    cols = ['trip_type', 'trv_type', 'room_type', 'days_number', 'submitted', 'pet']
    dateCols = ['rev_day', 'rev_month', 'rev_year']
    encodeColumns = ['Hotel_Name', 'Reviewer_Nationality', 'trip_type', 'room_type', 'trv_type', 'rev_day', 'rev_month', 'rev_year', 'days_number']

    df = pd.DataFrame()

    print("Processing Tags...")
    data[cols] = data[TAGS_COLUMN].apply(process_tags_column).apply(pd.Series)
    print("Tags Processed!")

    print("Processing Date...")
    data[dateCols] = data['Review_Date'].apply(fix_date_v2).apply(pd.Series)
    print("Date Processed!")
    # data[cols] = df[cols]
    # data[dateCols] = df[dateCols]

    print("Encoding Columns...")

    def encode_column(column):
        if column not in data.keys():
            return pd.Series(dtype='float64')
        encoded_column = labelEncoding(data[column]).astype('float64')
        return featureScaling(encoded_column)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(encode_column, encodeColumns))
    for i, column in enumerate(encodeColumns):
        df[column] = results[i]
    print("Columns Encoded!")
    return df
