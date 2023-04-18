import re

import pandas as pd
import requests
from src.Helpers import binarySearch, save, open_file, featureScaling

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
from src.Preprocessing import preprocessing, fix_date_v2

data = open_file("hotel-regression-dataset.csv")

# d = pd.DataFrame()
#
# d['name'] = []
# d['name'].append(1)
# print(d['name'])

# print(fix_date_v2(data['Review_Date'][0]))

df = preprocessing(data)
# print(df)

save(df, "test-preprocessing-v3.csv")

# for i in featureScaling(data['Additional_Number_of_Scoring'])[0]:
#     print(i)

# print(featureScaling(data['Additional_Number_of_Scoring']))

# d = data['Hotel_Address'][0]

# d = d[len(d)/2:]

# counter = 0
# for j in range(len(countries['country'])):
#     r = countries['country'][j].lower() in d.lower()
#     counter += 1
#     if r:
#         print(countries['country'][j].lower())
#         break
#
# print(counter)

# errors = dict()
# count = dict()
#
# for i in range(len(data['Hotel_Address'])):
#     for j in countries['country']:
#         s = j.lower()
#         if s in data['Hotel_Address'][i].lower():
#             if s not in count.keys():
#                 count[s] = 1
#             else:
#                 count[s] += 1
#         else:
#             errors[i] = data['Hotel_Address'][i].lower()
#
# print(count)
# print(errors)
