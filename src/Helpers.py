import re
import pickle
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from flashgeotext.geotext import GeoText, GeoTextConfiguration
from flashgeotext.lookup import LookupData
from requests.structures import CaseInsensitiveDict
from sklearn.model_selection import train_test_split
from config.constants import TARGET_COLUMN, CURRENT_VERSION



# Encoding
def encode(x, encoder, testing=False, label=True):
    en = encoder.transform(x) if testing else encoder.fit_transform(x)
    return en if label else pd.DataFrame(en.todense(), columns=encoder.get_feature_names())


def one_hot_encode(data, cols):
    """
    One-hot encode columns in a pandas DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to encode.
        cols (list): A list of column names to one-hot encode.

    Returns:
        A new DataFrame with the one-hot encoded columns.
    """
    # Create a OneHotEncoder object
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Fit the encoder on the specified columns
    encoder.fit(data[cols])

    # Transform the columns into a one-hot encoded array
    encoded_array = encoder.transform(data[cols]).toarray()

    # Create column names for the one-hot encoded columns
    # column_names = [f"{col}{category}" for col in cols for category in encoder.categories_[col]]

    # Create a new DataFrame with the one-hot encoded columns
    encoded_df = pd.DataFrame(encoded_array)

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    data = pd.concat([data, encoded_df], axis=1)

    # Drop the original columns
    data = data.drop(cols, axis=1)

    return data


def labelEncoding(x):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0, shuffle=False)
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

# fill lat & lng nulls
def geoCoding(text):
    url = f"https://api.geoapify.com/v1/geocode/search?apiKey=8a8befb5cff9493fa7f326a514a8d555&text={text}"

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"

    resp = requests.get(url, headers=headers)
    #print(resp.json()["features"][0]["geometry"]["coordinates"])
    coordinates = resp.json()["features"][0]["geometry"]["coordinates"]
    return coordinates[0], coordinates[1]

def getLatLng(data):
    d = data[data['lat'].isnull()]
    for index, row in d.iterrows():
        try:
            data.loc[index, ['lat', 'lng']] = geoCoding(row['Hotel_Address'])
        except Exception as e:
            print(data.isna().sum())
            print(f"stopped at - {index}, the error -> {e}")

    save(data,"cls-complete", CURRENT_VERSION)
    return data
##

# address preprocessing funs

# to init the dictionary of the existing countries.
def initcitiesDict():
    cities_dict = {
        'London': ['London'],
        'Manchester': ['Manchester'],
        'Birmingham': ['Birmingham'],
        'Liverpool': ['Liverpool'],
        'Leeds': ['Leeds'],
        'Glasgow': ['Glasgow'],
        'Edinburgh': ['Edinburgh'],
        'Bristol': ['Bristol'],
        'Newcastle upon Tyne': ['Newcastle upon Tyne'],
        'Sheffield': ['Sheffield'],
        'Nottingham': ['Nottingham'],
        'Belfast': ['Belfast'],
        'Southampton': ['Southampton'],
        'Brighton and Hove': ['Brighton and Hove'],
        'Cardiff': ['Cardiff'],
        'Paris': ['Paris'],
        'Marseille': ['Marseille'],
        'Lyon': ['Lyon'],
        'Toulouse': ['Toulouse'],
        'Nice': ['Nice'],
        'Nantes': ['Nantes'],
        'Strasbourg': ['Strasbourg'],
        'Montpellier': ['Montpellier'],
        'Bordeaux': ['Bordeaux'],
        'Lille': ['Lille'],
        'Rennes': ['Rennes'],
        'Reims': ['Reims'],
        'Le Havre': ['Le Havre'],
        'Saint-Étienne': ['Saint-Étienne'],
        'Toulon': ['Toulon'],
        'Grenoble': ['Grenoble'],
        'Dijon': ['Dijon'],
        'Angers': ['Angers'],
        'Nîmes': ['Nîmes'],
        'Villeurbanne': ['Villeurbanne'],
        'Rome': ['Rome'],
        'Milan': ['Milan'],
        'Naples': ['Naples'],
        'Turin': ['Turin'],
        'Palermo': ['Palermo'],
        'Genoa': ['Genoa'],
        'Bologna': ['Bologna'],
        'Florence': ['Florence'],
        'Bari': ['Bari'],
        'Catania': ['Catania'],
        'Venice': ['Venice'],
        'Verona': ['Verona'],
        'Messina': ['Messina'],
        'Padua': ['Padua'],
        'Trieste': ['Trieste'],
        'Taranto': ['Taranto'],
        'Brescia': ['Brescia'],
        'Prato': ['Prato'],
        'Reggio Calabria': ['Reggio Calabria'],
        'Modena': ['Modena'],
        'Madrid': ['Madrid'],
        'Barcelona': ['Barcelona'],
        'Valencia': ['Valencia'],
        'Seville': ['Seville'],
        'Malaga': ['Malaga'],
        'Bilbao': ['Bilbao'],
        'Murcia': ['Murcia'],
        'Palma de Mallorca': ['Palma de Mallorca'],
        'Las Palmas': ['Las Palmas'],
        'Zaragoza': ['Zaragoza'],
        'Alicante': ['Alicante'],
        'Cordoba': ['Cordoba'],
        'Valladolid': ['Valladolid'],
        'Vigo': ['Vigo'],
        'Gijon': ['Gijon'],
        'L Hospitalet de Llobregat': ['L Hospitalet de Llobregat'],
        'La Coruna': ['La Coruna'],
        'Granada': ['Granada'],
        'Cartagena': ['Cartagena'],
        'San Sebastian': ['San Sebastian'],
        'Amsterdam': ['Amsterdam'],
        'Rotterdam': ['Rotterdam'],
        'The Hague': ['The Hague'],
        'Den Haag': ['Den Haag'],
        'Utrecht': ['Utrecht'],
        'Eindhoven': ['Eindhoven'],
        'Tilburg': ['Tilburg'],
        'Groningen': ['Groningen'],
        'Almere': ['Almere'],
        'Breda': ['Breda'],
        'Nijmegen': ['Nijmegen'],
        'Apeldoorn': ['Apeldoorn'],
        'Haarlem': ['Haarlem'],
        'Enschede': ['Enschede'],
        'Arnhem': ['Arnhem'],
        'Amersfoort': ['Amersfoort'],
        'Vienna': ['Vienna'],
        'Graz': ['Graz'],
        'Linz': ['Linz'],
        'Salzburg': ['Salzburg'],
        'Innsbruck': ['Innsbruck'],
        'Klagenfurt': ['Klagenfurt'],
        'Villach': ['Villach'],
        'Wels': ['Wels'],
        'Sankt Pölten': ['Sankt Pölten'],
        'Dornbirn': ['Dornbirn'],
        'Wiener Neustadt': ['Wiener Neustadt'],
        'Steyr': ['Steyr'],
        'Feldkirch': ['Feldkirch'],
        'Bregenz': ['Bregenz'],
        'Leoben': ['Leoben']
    }
    lookup_districts = LookupData(
        name="city",
        data=cities_dict)
    config = GeoTextConfiguration(**{"use_demo_data": False})
    geotextCity = GeoText(config)
    geotextCity.add(lookup_districts)
    return geotextCity


def GetCity(a, geotextCity):
    c = geotextCity.extract(a['Hotel_Address'], span_info=True)
    return list(c['city'].keys())[-1]


def GetCountry(a, geotextCountry):
    n = geotextCountry.extract(a['Hotel_Address'], span_info=True)
    return list(n['countries'].keys())[-1]

##

def extractNumberFromString(text):
    res = re.search('[0-9]+', text)
    return res.group() if res else None
