import pandas as pd
from Preprocessing import process_tags_column, processLongLat
from src.Helpers import save

data = pd.read_csv("../input/hotel-regression-dataset.csv")


def main():
    data[["Trip_Type", "Travel_Type", "Room_Type", "Days_Number", "Submitted_From_Mobile", "With_Pet", "Remaining"]] = data['Tags'].apply(process_tags_column).apply(pd.Series)

    data[["lat", "lng"]] = data[data['lat'].isnull()].iloc[0:3, :].apply(processLongLat, axis=1).apply(pd.Series)

    save(data, 'hotels-with-add', 3)

    data2 = pd.read_csv("../input/hotels-with-add-3.csv")

    print(data2.isna().sum())


if __name__ == "__main__":
    main()
