import re

import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from requests.structures import CaseInsensitiveDict


# Encoding
def labelEncoding(x):
    # if x.dtype != 'float64':
    #     return x
    lbl = LabelEncoder()
    return lbl.fit_transform(x)


def oneHotEncoding(x):
    return x


# Scaling
def featureScaling(x):
    mini, maxi = min(x), max(x)

    if mini == maxi:
        return x

    x = (x - mini) / (maxi - mini)

    return x
    # scaler = MinMaxScaler()
    # # fit and transform the column
    # return scaler.fit_transform([x])


def save(data, fileName, v, f='csv'):
    try:
        data.to_csv(f"../input/{fileName}-v{v}.{f}")
    except:
        save(data, fileName, v + 1, f)


def open_file(fileName):
    return pd.read_csv(f"../input/{fileName}")


def binarySearch(arr, x):
    low = 0
    high = len(arr) - 1

    while low <= high:

        mid = (high + low) // 2

        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1

        # If x is smaller, ignore right half
        elif arr[mid] > x:
            high = mid - 1

        # means x is present at mid
        else:
            return mid

    # If we reach here, then the element was not present
    return -1


def geoCoding(text):
    url = f"https://api.geoapify.com/v1/geocode/search?apiKey=8a8befb5cff9493fa7f326a514a8d555&text={text}"

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"

    resp = requests.get(url, headers=headers)

    return resp.json()


def extractNumberFromString(text):
    res = re.search('[0-9]+', text)
    return res.group() if res else None
