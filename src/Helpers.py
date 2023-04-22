import re
import pickle
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from requests.structures import CaseInsensitiveDict


# Encoding
def labelEncoding(x, encoder=None):
    if encoder is not None:
        return encoder.fit_transform(x)
    encoder = LabelEncoder()
    return encoder, encoder.fit_transform(x)


def oneHotEncoding(x, encoder=None):
    if encoder is not None:
        encoded = encoder.fit_transform(x)
        return pd.DataFrame(encoded.todense(), columns=encoder.get_feature_names())
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(x)
    return encoder, pd.DataFrame(encoded.todense(), columns=encoder.get_feature_names())


# Scaling
def featureScaling(x):
    mini, maxi = min(x), max(x)

    if mini == maxi:
        return x

    x = (x - mini) / (maxi - mini)

    return x


def save(data, fileName, v, f='csv'):
    try:
        data.to_csv(f"../input/{fileName}-v{v}.{f}")
        print(f"Last version is {v}")
    except:
        save(data, fileName, v + 1, f)


def open_file(fileName, path="../input/"):
    return pd.read_csv(f"{path}{fileName}")


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


def pickleStore(data, name):
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(data, f)


def pickleOpen(name):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)
