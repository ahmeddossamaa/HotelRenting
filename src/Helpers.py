import re
import pickle
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from requests.structures import CaseInsensitiveDict
from sklearn.model_selection import train_test_split


# Encoding
def encode(x, encoder, testing=False, label=True):
    en = encoder.transform(x) if testing else encoder.fit_transform(x)
    return en if label else pd.DataFrame(en.todense(), columns=encoder.get_feature_names())


def labelEncoding(x, encoder=None):
    if encoder is not None:
        return encoder.fit_transform(x)
    encoder = LabelEncoder()
    return encoder, encoder.fit_transform(x)


def oneHotEncoding(x, encoder=None):
    if encoder is not None:
        encoded = encoder.fit_transform(x)
        return pd.DataFrame(encoded.todense(), columns=encoder.get_feature_names())
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded = encoder.fit_transform(x)
    return encoder, pd.DataFrame(encoded.todense(), columns=encoder.get_feature_names())


##

# Scaling
def featureScaling(x):
    mini, maxi = min(x), max(x)

    if mini == maxi:
        return x

    x = (x - mini) / (maxi - mini)

    return x


def featureScalingScikit(df):
    scaler = StandardScaler()

    # Scale the featuress
    X_scaled = scaler.fit_transform(df)
    scaled = pd.DataFrame(X_scaled, columns=df.columns)
    return scaled


##

# splitting dataset into train, valid, test
def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, shuffle=True)
    return X_train, X_test, X_val, y_train, y_test, y_val


##

# Save and Open CSV files
def save(data, fileName, v, f='csv'):
    try:
        data.to_csv(f"../input/{fileName}-v{v}.{f}", index=False)
        print(f"Last version is {v}")
    except:
        save(data, fileName, v + 1, f)


def open_file(fileName, path="../input/"):
    return pd.read_csv(f"{path}{fileName}")


##

# Save & Open ml objects
def pickleStore(data, name):
    try:
        with open(f'../models/{name}.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(f"{name} saved in models")
    except Exception as e:
        print(f"{name} couldn't be saved: {e}")


def pickleOpen(name):
    try:
        with open(f'../models/{name}.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error fetching {name}: {e}")


##


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
