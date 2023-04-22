import re

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from config.constants import TAGS_COLUMN, ADDRESS_COLUMN, TARGET_COLUMN
from src.Helpers import binarySearch, save, open_file, featureScaling, pickleOpen

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
from src.Model import logisticRegression
from src.Preprocessing import preprocessing

# data = open_file("Hotel_Renting_v4.csv")
# data = open_file("Hotel_Renting_v4.csv")
#
# X = data.loc[:, data.columns != TARGET_COLUMN]
# y = data[TARGET_COLUMN]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# #
# df_training = pd.concat([X_test, y_test], axis=1)
# #
# df_training = preprocessing(df_training)
# print([i for i in X_test])
# print(pickleOpen("encoders")['trv_type'].transform(X_test['trv_type']))

logisticRegression()
