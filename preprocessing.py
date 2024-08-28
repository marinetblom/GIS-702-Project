# preprocessing.py
import pandas as pd
import numpy as np

################################


def load_cape_town():
    # Load the Cape Town dataset
    df = pd.read_csv(
        r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Data\0021178a3 5min.ttx",
        delimiter="\t",
        encoding="latin1",
        decimal=",",
        parse_dates=["DateT"],
    )
    # Clean the Pressure column
    df["Pressure"] = df["Pressure"].apply(clean_pressure)
    df["Pressure"] = df["Pressure"].astype("float64")

    # Filter the dataframe to exclude 1997 and include only the desired years
    filtered_df = df[
        (df["DateT"].dt.year >= 1995) & (df["DateT"].dt.year <= 1998)
        | (df["DateT"].dt.year >= 2001) & (df["DateT"].dt.year <= 2023)
    ]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def clean_pressure(value):
    if isinstance(value, str):
        cleaned_value = "".join(c for c in value if c.isdigit() or c == "." or c == ",")
        cleaned_value = cleaned_value.replace(",", ".")
        try:
            return float(cleaned_value)
        except ValueError:
            return np.nan


################################


def load_bloemfontein():
    # Load the Bloemfontein dataset
    df = pd.read_csv(
        r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Data\0261516b0 5min.ttx",
        delimiter="\t",
        encoding="latin1",
        decimal=",",
        parse_dates=["DateT"],
    )
    # Filter the dataframe to exclude 1997 and include only the desired years
    filtered_df = df[
        (df["DateT"].dt.year >= 1995) & (df["DateT"].dt.year <= 1996)
        | (df["DateT"].dt.year >= 1998) & (df["DateT"].dt.year <= 2023)
    ]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df
