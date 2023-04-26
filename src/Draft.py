import pandas as pd

from config.constants import CURRENT_VERSION, ADDRESS_COLUMN
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

# data = enc
# d = dict()
# i = "test"
# print(d[i] if i in d.keys() else "haha")

# data = open_file("hotel-regression-dataset.csv")
# print(f"res = {getCity(data[ADDRESS_COLUMN][0])}")

import locationtagger

data = open_file("hotel-regression-dataset.csv")
text = data[ADDRESS_COLUMN][9]
print(text)
place_entity = locationtagger.find_locations(text=text)

print(place_entity.countries)
print(place_entity.cities)
print(place_entity.regions)
print(place_entity.region_cities)
print(place_entity.other_regions)
print(place_entity.other)


# import nltk
# import spacy
#
# # essential entity models downloads
# nltk.downloader.download('maxent_ne_chunker')
# nltk.downloader.download('words')
# nltk.downloader.download('treebank')
# nltk.downloader.download('maxent_treebank_pos_tagger')
# nltk.downloader.download('punkt')
# nltk.download('averaged_perceptron_tagger')
