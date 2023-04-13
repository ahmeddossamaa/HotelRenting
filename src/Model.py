import re

import pandas as pd
from Preprocessing import process_tags_column_v2, save

data = pd.read_csv("../input/hotel-regression-dataset.csv")


def main():
    data[["Trip_Type", "Travel_Type", "Room_Type", "Days_Number"]] = data['Tags'].apply(process_tags_column_v2).apply(pd.Series)

    save(data, 'hotels-with-add.csv')

    data2 = pd.read_csv("../input/hotels-with-add.csv")

    print(data2.isna().sum())


# arr = eval(data['Tags'][0])
#
# print(arr[0].lower().strip())
# print(re.search("trip", arr[0].lower().strip()))

main()

# print(len(data['Hotel_Name'].unique()))
