import pandas as pd


def get_csv_data(path):
    """Reads in CSV file exported from SQL db"""
    df = pd.read_csv(path, index_col="time")
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df
