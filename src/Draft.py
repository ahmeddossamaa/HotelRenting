import re

import pandas as pd
import requests
from src.Helpers import binarySearch, save, open_file

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

countries = open_file("countries-cities.csv")
data = open_file("hotel-regression-dataset.csv")

d = data['Hotel_Address'][0]

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

errors = dict()
count = dict()

for i in range(len(data['Hotel_Address'])):
    for j in countries['country']:
        s = j.lower()
        if s in data['Hotel_Address'][i].lower():
            if s not in count.keys():
                count[s] = 1
            else:
                count[s] += 1
        else:
            errors[i] = data['Hotel_Address'][i].lower()

print(count)
print(errors)
