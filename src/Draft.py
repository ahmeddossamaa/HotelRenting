import pandas as pd

from config.constants import CURRENT_VERSION
from src.Helpers import save, open_file
from src.Preprocessing import GetMissingTripType

# from config.constants import TAGS_COLUMN, ADDRESS_COLUMN, TARGET_COLUMN
# from src.Helpers import binarySearch, save, open_file, featureScaling, pickleOpen

# URL = "https://countriesnow.space/api/v0.1/countries"
#
# r = requests.get(URL)
# rows = r.json()['data']
#
# print(rows)

# df = pd.DataFrame(rows)
#
# save(df, "countries-cities.csv")
#
# for i in df['country']:
#     print(i)

# countries = open_file("countries-cities.csv")

# from src.Preprocessing import preprocessing

# data = open_file("Hotel_Renting_v4.csv")
# data = open_file("Hotel_Renting_v4.csv")
#
# X = data.loc[:, data.columns != TARGET_COLUM"N]
# y = data[TARGET_COLUMN]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# #
# df_training = pd.concat([X_test, y_test], axis=1)
# #
# df_training = preprocessing(df_training)
# print([i for i in X_test])
# print(pickleOpen("encoders")['trv_type'].transform(X_test['trv_type']))

# data = open_file("columns-processing-v1.csv")
# data = data.iloc[:, 2:]
# d = GetMissingTripType(data)
#
# save(data, "hotel-dataset-processed", CURRENT_VERSION)
# data.to_csv("proccesed_col-v1.csv")
from flashgeotext.geotext import GeoText, GeoTextConfiguration
from flashgeotext.lookup import LookupData

# import geograpy
# from geotext import GeoT
from flashgeotext.geotext import GeoText, GeoTextConfiguration
from flashgeotext.lookup import LookupData

# import geograpy
# from geotext import GeoT
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

print(cities_dict)

lookup_districts = LookupData(
    name="city",
    data=cities_dict)

config = GeoTextConfiguration(**{"use_demo_data": False})
geotext = GeoText(config)
geotext.add(lookup_districts)
print(len(geotext.pool))
# from src.Helpers import binarySearch, save, open_file

# URL = "https://countriesnow.space/api/v0.1/countries"
#
# r = requests.get(URL)
# rows = r.json()['data']
#
# print(rows)

# df = pd.DataFrame(rows)
#
# save(df, "countries-cities.csv")
#
# for i in df['country']:
#     print(i)

# countries = open_file("countries-cities.csv")

data = pd.read_csv("C:/Users/GH/Documents/HotelRenting/input/Hotel_Renting_v4.csv")
data = data.drop(data.columns[0], axis=1)

print(data['Hotel_Country'].unique())


def proccess(a):
    # print(a['Hotel_Address'])
    out = geotext.extract(a['Hotel_Address'], span_info=True)
    # print(out)
    # print(f"city = {list(out['cities'].keys())}, country ={list(out['countries'].keys())[0]} ")
    return list(out['city'].keys())[-1]


data['Hotel_City'] = data.apply(proccess, axis=1)

print(data['Hotel_City'].unique())
# from src.Helpers import binarySearch, save, open_file

# URL = "https://countriesnow.space/api/v0.1/countries"
#
# r = requests.get(URL)
# rows = r.json()['data']
#
# print(rows)

# df = pd.DataFrame(rows)
#
# save(df, "countries-cities.csv")
#
# for i in df['country']:
#     print(i)

# countries = open_file("countries-cities.csv")

data = pd.read_csv("Hotel_Renting_v3.csv")
data = data.drop(data.columns[0], axis=1)

print(data['Hotel_Country'].unique())


def proccess(a):
    # print(a['Hotel_Address'])
    out = geotext.extract(a['Hotel_Address'], span_info=True)
    # print(out)
    # print(f"city = {list(out['cities'].keys())}, country ={list(out['countries'].keys())[0]} ")
    return list(out['city'].keys())[-1]


data['Hotel_City'] = data.apply(proccess, axis=1)

print(data['Hotel_City'].unique())
# data = enc
# d = dict()
# i = "test"
# print(d[i] if i in d.keys() else "haha")
