import concurrent.futures

import pandas as pd
from sklearn.model_selection import train_test_split

from config.constants import CURRENT_VERSION, ADDRESS_COLUMN, TARGET_COLUMN, TAGS_COLUMN, DATE_COLUMN, \
    REGRESSION_DATASET
from src.Helpers import save, open_file
from src.Preprocessing import GetMissingTripType, encodeColumns, getCityAndCountry

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
# from flashgeotext.geotext import GeoText, GeoTextConfiguration
# from flashgeotext.lookup import LookupData
#
# # import geograpy
# # from geotext import GeoT
# from flashgeotext.geotext import GeoText, GeoTextConfiguration
# from flashgeotext.lookup import LookupData
#
# # import geograpy
# # from geotext import GeoT
# cities_dict = {
#     'London': ['London'],
#     'Manchester': ['Manchester'],
#     'Birmingham': ['Birmingham'],
#     'Liverpool': ['Liverpool'],
#     'Leeds': ['Leeds'],
#     'Glasgow': ['Glasgow'],
#     'Edinburgh': ['Edinburgh'],
#     'Bristol': ['Bristol'],
#     'Newcastle upon Tyne': ['Newcastle upon Tyne'],
#     'Sheffield': ['Sheffield'],
#     'Nottingham': ['Nottingham'],
#     'Belfast': ['Belfast'],
#     'Southampton': ['Southampton'],
#     'Brighton and Hove': ['Brighton and Hove'],
#     'Cardiff': ['Cardiff'],
#     'Paris': ['Paris'],
#     'Marseille': ['Marseille'],
#     'Lyon': ['Lyon'],
#     'Toulouse': ['Toulouse'],
#     'Nice': ['Nice'],
#     'Nantes': ['Nantes'],
#     'Strasbourg': ['Strasbourg'],
#     'Montpellier': ['Montpellier'],
#     'Bordeaux': ['Bordeaux'],
#     'Lille': ['Lille'],
#     'Rennes': ['Rennes'],
#     'Reims': ['Reims'],
#     'Le Havre': ['Le Havre'],
#     'Saint-Étienne': ['Saint-Étienne'],
#     'Toulon': ['Toulon'],
#     'Grenoble': ['Grenoble'],
#     'Dijon': ['Dijon'],
#     'Angers': ['Angers'],
#     'Nîmes': ['Nîmes'],
#     'Villeurbanne': ['Villeurbanne'],
#     'Rome': ['Rome'],
#     'Milan': ['Milan'],
#     'Naples': ['Naples'],
#     'Turin': ['Turin'],
#     'Palermo': ['Palermo'],
#     'Genoa': ['Genoa'],
#     'Bologna': ['Bologna'],
#     'Florence': ['Florence'],
#     'Bari': ['Bari'],
#     'Catania': ['Catania'],
#     'Venice': ['Venice'],
#     'Verona': ['Verona'],
#     'Messina': ['Messina'],
#     'Padua': ['Padua'],
#     'Trieste': ['Trieste'],
#     'Taranto': ['Taranto'],
#     'Brescia': ['Brescia'],
#     'Prato': ['Prato'],
#     'Reggio Calabria': ['Reggio Calabria'],
#     'Modena': ['Modena'],
#     'Madrid': ['Madrid'],
#     'Barcelona': ['Barcelona'],
#     'Valencia': ['Valencia'],
#     'Seville': ['Seville'],
#     'Malaga': ['Malaga'],
#     'Bilbao': ['Bilbao'],
#     'Murcia': ['Murcia'],
#     'Palma de Mallorca': ['Palma de Mallorca'],
#     'Las Palmas': ['Las Palmas'],
#     'Zaragoza': ['Zaragoza'],
#     'Alicante': ['Alicante'],
#     'Cordoba': ['Cordoba'],
#     'Valladolid': ['Valladolid'],
#     'Vigo': ['Vigo'],
#     'Gijon': ['Gijon'],
#     'L Hospitalet de Llobregat': ['L Hospitalet de Llobregat'],
#     'La Coruna': ['La Coruna'],
#     'Granada': ['Granada'],
#     'Cartagena': ['Cartagena'],
#     'San Sebastian': ['San Sebastian'],
#     'Amsterdam': ['Amsterdam'],
#     'Rotterdam': ['Rotterdam'],
#     'The Hague': ['The Hague'],
#     'Den Haag': ['Den Haag'],
#     'Utrecht': ['Utrecht'],
#     'Eindhoven': ['Eindhoven'],
#     'Tilburg': ['Tilburg'],
#     'Groningen': ['Groningen'],
#     'Almere': ['Almere'],
#     'Breda': ['Breda'],
#     'Nijmegen': ['Nijmegen'],
#     'Apeldoorn': ['Apeldoorn'],
#     'Haarlem': ['Haarlem'],
#     'Enschede': ['Enschede'],
#     'Arnhem': ['Arnhem'],
#     'Amersfoort': ['Amersfoort'],
#     'Vienna': ['Vienna'],
#     'Graz': ['Graz'],
#     'Linz': ['Linz'],
#     'Salzburg': ['Salzburg'],
#     'Innsbruck': ['Innsbruck'],
#     'Klagenfurt': ['Klagenfurt'],
#     'Villach': ['Villach'],
#     'Wels': ['Wels'],
#     'Sankt Pölten': ['Sankt Pölten'],
#     'Dornbirn': ['Dornbirn'],
#     'Wiener Neustadt': ['Wiener Neustadt'],
#     'Steyr': ['Steyr'],
#     'Feldkirch': ['Feldkirch'],
#     'Bregenz': ['Bregenz'],
#     'Leoben': ['Leoben']
# }
#
# print(cities_dict)
#
# lookup_districts = LookupData(
#     name="city",
#     data=cities_dict)
#
# config = GeoTextConfiguration(**{"use_demo_data": False})
# geotext = GeoText(config)
# geotext.add(lookup_districts)
# print(len(geotext.pool))
# # from src.Helpers import binarySearch, save, open_file
#
# # URL = "https://countriesnow.space/api/v0.1/countries"
# #
# # r = requests.get(URL)
# # rows = r.json()['data']
# #
# # print(rows)
#
# # df = pd.DataFrame(rows)
# #
# # save(df, "countries-cities.csv")
# #
# # for i in df['country']:
# #     print(i)
#
# # countries = open_file("countries-cities.csv")
#
# data = pd.read_csv("C:/Users/GH/Documents/HotelRenting/input/Hotel_Renting_v4.csv")
# data = data.drop(data.columns[0], axis=1)
#
# print(data['Hotel_Country'].unique())
#
#
# def proccess(a):
#     # print(a['Hotel_Address'])
#     out = geotext.extract(a['Hotel_Address'], span_info=True)
#     # print(out)
#     # print(f"city = {list(out['cities'].keys())}, country ={list(out['countries'].keys())[0]} ")
#     return list(out['city'].keys())[-1]
#
#
# data['Hotel_City'] = data.apply(proccess, axis=1)
#
# print(data['Hotel_City'].unique())
# # from src.Helpers import binarySearch, save, open_file
#
# # URL = "https://countriesnow.space/api/v0.1/countries"
# #
# # r = requests.get(URL)
# # rows = r.json()['data']
# #
# # print(rows)
#
# # df = pd.DataFrame(rows)
# #
# # save(df, "countries-cities.csv")
# #
# # for i in df['country']:
# #     print(i)
#
# # countries = open_file("countries-cities.csv")
#
# data = pd.read_csv("Hotel_Renting_v3.csv")
# data = data.drop(data.columns[0], axis=1)
#
# print(data['Hotel_Country'].unique())
#
#
# def proccess(a):
#     # print(a['Hotel_Address'])
#     out = geotext.extract(a['Hotel_Address'], span_info=True)
#     # print(out)
#     # print(f"city = {list(out['cities'].keys())}, country ={list(out['countries'].keys())[0]} ")
#     return list(out['city'].keys())[-1]
#
#
# data['Hotel_City'] = data.apply(proccess, axis=1)
#
# print(data['Hotel_City'].unique())
# # data = enc
# # d = dict()
# # i = "test"
# # print(d[i] if i in d.keys() else "haha")
#
# # data = open_file("hotel-regression-dataset.csv")
# # print(f"res = {getCity(data[ADDRESS_COLUMN][0])}")
#
# import locationtagger
#
# data = open_file("hotel-regression-dataset.csv")
# text = data[ADDRESS_COLUMN][9]
# print(text)
# place_entity = locationtagger.find_locations(text=text)
#
# print(place_entity.countries)
# print(place_entity.cities)
# print(place_entity.regions)
# print(place_entity.region_cities)
# print(place_entity.other_regions)
# print(place_entity.other)
#
#
# # import nltk
# # import spacy
# #
# # # essential entity models downloads
# # nltk.downloader.download('maxent_ne_chunker')
# # nltk.downloader.download('words')
# # nltk.downloader.download('treebank')
# # nltk.downloader.download('maxent_treebank_pos_tagger')
# # nltk.downloader.download('punkt')
# # nltk.download('averaged_perceptron_tagger')

# data = open_file("hotel-regression-dataset.csv")
#
# X = data.loc[:, data.columns != TARGET_COLUMN]
# y = data[TARGET_COLUMN]
#
# xtr, xts, ytr, yts = train_test_split(X, y, test_size=0.01)
#
# data = pd.concat([xts, yts], axis=1)
#
# print(data['Hotel_City'].unique())
# data = enc
# d = dict()
# i = "test"
# print(d[i] if i in d.keys() else "haha")



# to handle test nulls -> check for each col? if that col isn't one of the selected drop col otherwise trytohandle!

# data = open_file("processed-columns-v1.csv")
#
# data = encodeColumns(data, {
#     'days_number': {
#         'label': True,
#         'oneHot': False,
#     }
# }, file="testingDays")
#
# save(data, "testingDays", CURRENT_VERSION)

data = open_file(REGRESSION_DATASET)

data = getCityAndCountry(data)

save(data, "getCityAndCountry", 1)
