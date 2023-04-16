import re
import pandas as pd

from Preprocessing import process_tags_column, processLongLat
from config.constants import ADDRESS_COLUMN
from src.Helpers import geoCoding, save

data = pd.read_csv("../input/hotel-regression-dataset.csv")


def main():
    data[["Trip_Type", "Travel_Type", "Room_Type", "Days_Number", "Submitted_From_Mobile", "With_Pet", "Remaining"]] = data['Tags'].apply(process_tags_column).apply(pd.Series)
    # d = data[data['lat'].isnull()].iloc[0:3, :]
    # print(d)
    # for i in d:
    #     print(i)
    data[["lat", "lng"]] = data[data['lat'].isnull()].iloc[0:3, :].apply(processLongLat, axis=1).apply(pd.Series)

    save(data, 'hotels-with-add-3.csv')

    data2 = pd.read_csv("../input/hotels-with-add-3.csv")

    print(data2.isna().sum())


main()

# d = data[data['lat'].isnull()].iloc[0:3, :]
# print("d= ", d)
# obj = d.apply(processLongLat, axis=1).apply(pd.Series)
# print(obj)


# d = data[data['lat'].isnull()]
# print(d)
#
# newCoordinates = dict()
# for i in range(len(d)):
#     res = geoCoding(d.iloc[i][ADDRESS_COLUMN])
#     if res:
#         res = res['features'][0]['geometry']['coordinates']
#         newCoordinates[i] = {
#             'lat': res[1],
#             'lng': res[0],
#         }
#
# print(newCoordinates)

# print(data.loc[:, 14:16])
# res = geoCoding(data[ADDRESS_COLUMN][0])['features'][0]['geometry']['coordinates']
# print(f"lat={res[1]}, lng={res[0]}")

"""{'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [ 
-0.151015, 51.510608]}, 'properties': {'country_code': 'gb', 'housenumber': '44', 'street': 'Grosvenor Square', 
'country': 'United Kingdom', 'datasource': {'sourcename': 'openstreetmap', 'attribution': '© OpenStreetMap 
contributors', 'license': 'Open Database License', 'url': 'https://www.openstreetmap.org/copyright'}, 'postcode': 
'W1K 2HP', 'state': 'England', 'city': 'London', 'district': 'Westminster', 'suburb': 'Mayfair', 'county': 'Greater 
London', 'lon': -0.151015, 'lat': 51.510608, 'state_code': 'ENG', 'formatted': '44 Grosvenor Square, London, W1K 2HP, 
United Kingdom', 'address_line1': '44 Grosvenor Square', 'address_line2': 'London, W1K 2HP, United Kingdom', 
'timezone': {'name': 'Europe/London', 'offset_STD': '+00:00', 'offset_STD_seconds': 0, 'offset_DST': '+01:00', 
'offset_DST_seconds': 3600, 'abbreviation_STD': 'GMT', 'abbreviation_DST': 'BST'}, 'result_type': 'building', 
'rank': {'popularity': 8.685225217471704, 'confidence': 0.5, 'confidence_city_level': 1, 'confidence_street_level': 
0.5, 'match_type': 'full_match'}, 'place_id': 
'51dc4b1aa37554c3bf59b9895a9a5bc14940f00102f90157b3480200000000c00203e203226f70656e7374726565746d61703a616464726573733a7761792f3338333138393335'}}, {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-0.15152, 51.510441]}, 'properties': {'country_code': 'gb', 'housenumber': '38', 'street': 'Grosvenor Square', 'country': 'United Kingdom', 'datasource': {'sourcename': 'openstreetmap', 'attribution': '© OpenStreetMap contributors', 'license': 'Open Database License', 'url': 'https://www.openstreetmap.org/copyright'}, 'state': 'England', 'city': 'London', 'district': 'Westminster', 'suburb': 'Mayfair', 'county': 'Greater London', 'lon': -0.15152, 'lat': 51.510441, 'state_code': 'ENG', 'postcode': 'W1K 2HP', 'formatted': '38 Grosvenor Square, London, W1K 2HP, United Kingdom', 'address_line1': '38 Grosvenor Square', 'address_line2': 'London, W1K 2HP, United Kingdom', 'timezone': {'name': 'Europe/London', 'offset_STD': '+00:00', 'offset_STD_seconds': 0, 'offset_DST': '+01:00', 'offset_DST_seconds': 3600, 'abbreviation_STD': 'GMT', 'abbreviation_DST': 'BST'}, 'result_type': 'building', 'rank': {'popularity': 8.618703219718846, 'confidence': 0.35, 'confidence_city_level': 1, 'confidence_street_level': 0.5, 'match_type': 'full_match'}, 'place_id': '514c4f58e20165c3bf59cec4742156c14940f00102f901f2146a1400000000c00203e203236f70656e7374726565746d61703a616464726573733a7761792f333432343936343938'}}, {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-0.151772, 51.5103]}, 'properties': {'country_code': 'gb', 'housenumber': '35-37', 'street': 'Grosvenor Square', 'country': 'United Kingdom', 'datasource': {'sourcename': 'openstreetmap', 'attribution': '© OpenStreetMap contributors', 'license': 'Open Database License', 'url': 'https://www.openstreetmap.org/copyright'}, 'state': 'England', 'city': 'London', 'district': 'Westminster', 'suburb': 'Mayfair', 'county': 'Greater London', 'lon': -0.151772, 'lat': 51.5103, 'state_code': 'ENG', 'postcode': 'W1K 2HP', 'formatted': '35-37 Grosvenor Square, London, W1K 2HP, United Kingdom', 'address_line1': '35-37 Grosvenor Square', 'address_line2': 'London, W1K 2HP, United Kingdom', 'timezone': {'name': 'Europe/London', 'offset_STD': '+00:00', 'offset_STD_seconds': 0, 'offset_DST': '+01:00', 'offset_DST_seconds': 3600, 'abbreviation_STD': 'GMT', 'abbreviation_DST': 'BST'}, 'result_type': 'building', 'rank': {'popularity': 8.618703219718846, 'confidence': 0.3333333333333333, 'confidence_city_level': 1, 'confidence_street_level': 0.5, 'match_type': 'full_match'}, 'place_id': '51d76839d0436dc3bf590c93a98251c14940f00102f9014198670d00000000c00203e203236f70656e7374726565746d61703a616464726573733a7761792f323234383932393933'}}, {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-0.149807, 51.511637]}, 'properties': {'country_code': 'gb', 'housenumber': '3', 'street': 'Grosvenor Square', 'country': 'United Kingdom', 'datasource': {'sourcename': 'openstreetmap', 'attribution': '© OpenStreetMap contributors', 'license': 'Open Database License', 'url': 'https://www.openstreetmap.org/copyright'}, 'state': 'England', 'city': 'London', 'district': 'Westminster', 'suburb': 'Mayfair', 'county': 'Greater London', 'lon': -0.149807, 'lat': 51.511637, 'state_code': 'ENG', 'postcode': 'W1K 2HP', 'formatted': '3 Grosvenor Square, London, W1K 2HP, United Kingdom', 'address_line1': '3 Grosvenor Square', 'address_line2': 'London, W1K 2HP, United Kingdom', 'timezone': {'name': 'Europe/London', 'offset_STD': '+00:00', 'offset_STD_seconds': 0, 'offset_DST': '+01:00', 'offset_DST_seconds': 3600, 'abbreviation_STD': 'GMT', 'abbreviation_DST': 'BST'}, 'result_type': 'building', 'rank': {'popularity': 8.802208760464808, 'confidence': 0.25, 'confidence_city_level': 1, 'confidence_street_level': 0.5, 'match_type': 'full_match'}, 'place_id': '519f1edb32e02cc3bf5937363b527dc14940f00103f90120f2891b00000000c00203e203246f70656e7374726565746d61703a616464726573733a6e6f64652f343632303235323438'}}, {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-0.149259, 51.511622]}, 'properties': {'country_code': 'gb', 'housenumber': '1', 'street': 'Grosvenor Square', 'country': 'United Kingdom', 'datasource': {'sourcename': 'openstreetmap', 'attribution': '© OpenStreetMap contributors', 'license': 'Open Database License', 'url': 'https://www.openstreetmap.org/copyright'}, 'state': 'England', 'city': 'London', 'district': 'Westminster', 'suburb': 'Mayfair', 'county': 'Greater London', 'lon': -0.149259, 'lat': 51.511622, 'state_code': 'ENG', 'postcode': 'W1K 2HP', 'formatted': '1 Grosvenor Square, London, W1K 2HP, United Kingdom', 'address_line1': '1 Grosvenor Square', 'address_line2': 'London, W1K 2HP, United Kingdom', 'timezone': {'name': 'Europe/London', 'offset_STD': '+00:00', 'offset_STD_seconds': 0, 'offset_DST': '+01:00', 'offset_DST_seconds': 3600, 'abbreviation_STD': 'GMT', 'abbreviation_DST': 'BST'}, 'result_type': 'building', 'rank': {'popularity': 8.830562398851272, 'confidence': 0.25, 'confidence_city_level': 1, 'confidence_street_level': 0.5, 'match_type': 'full_match'}, 'place_id': '51e71bd13deb1ac3bf5902f566d47cc14940f00102f9016b619e2400000000c00203e203236f70656e7374726565746d61703a616464726573733a7761792f363134333539343033'}}], 'query': {'text': '44 Grosvenor Square Westminster Borough London W1K 2HP United Kingdom', 'parsed': {'housenumber': '44', 'street': 'grosvenor square westminster borough', 'postcode': 'w1k 2hp', 'city': 'london', 'country': 'united kingdom', 'expected_type': 'building'}}} """
