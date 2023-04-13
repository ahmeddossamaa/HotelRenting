import re
import datetime as dt
import pandas as pd


def save(data, fileName):
    data.to_csv(f"../input/{fileName}")


def fix_date(x):
    for i in range(len(x)):
        if x[i].find("-") != -1:
            t = x[i].split("-")
        elif x[i].find("/") != -1:
            t = x[i].split("/")
        else:
            continue

        x[i] = float(dt.date(int(t[2]), int(t[1]), int(t[0])).timetuple().tm_yday)

    return pd.DataFrame({'Date': x}, dtype='float64')


def fix_date_v2(data):
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True, dayfirst=True)
    data.info()
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data = data.drop('Date', axis=1)
    data.info()


def process_tags_Column(a):
    b = eval(a)

    trip_type = ''
    trv_type = ''
    room_type = ''
    days_number = ''

    if len(b) > 0:
        if 'trip' in b[0]:
            trip_type = b[0].strip('[]\' ')
        else:
            types = ['group', 'couple', 'solo', 'family']
            for t in types:
                if t in b[0].lower():
                    trv_type = b[0].strip('[]\' ')
        if len(b) > 1:
            types = ['group', 'couple', 'solo', 'family']
            for t in types:
                if t in b[1].lower():
                    trv_type = b[1].strip('[]\' ')
            if trv_type == '':
                if 'room' or 'bed' in b[1].lower():
                    room_type = b[1].strip('[]\' ')
        if len(b) > 2:
            if 'room' or 'bed' in b[2].lower():
                room_type = b[2].strip('[]\' ')
            else:
                if 'night' in b[2].lower():
                    days_number = b[2].strip('[]\' ')
        if len(b) > 3:
            if 'night' in b[3].lower():
                days_number = b[3].strip('[]\' ')

    # print(f'trip = {trip_type}, {trv_type}, {room_type}, {days_number}')
    return trip_type, trv_type, room_type, days_number


def process_tags_column_v2(a):
    b = eval(a)

    trip_type = ''
    trv_type = ''
    room_type = ''
    days_number = ''
    submission = ''

    for i in b:
        i = i.lower().strip()

        check = lambda r: re.search(r, i)

        if check(r"\btrip\b"):
            trip_type = i
        elif check(r"group|couple|solo|family"):
            trv_type = i
        elif check(r"room[s]?|bed[s]?"):
            room_type = i
        elif check(r"night[s]?"):
            days_number = i

    return trip_type, trv_type, room_type, days_number
