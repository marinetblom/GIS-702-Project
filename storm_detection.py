# storm_detection.py
import pandas as pd
import matplotlib.pyplot as plt


# Function to identify annual maximum rainfall
def identify_annual_max_rainfall(df):
    return df.loc[df.groupby(df["DateT"].dt.year)["Rain"].idxmax()]


# Function to plot annual maximum rainfall
def plot_annual_max_rainfall(df):
    # Extract the year from the 'DateT' column
    df["Year"] = df["DateT"].dt.year

    # Plot the annual maximum rainfall
    plt.figure(figsize=(12, 6))
    plt.bar(df["Year"], df["Rain"], color="skyblue")
    plt.title("Annual Maximum Rainfall (5-min interval)")
    plt.xlabel("Year")
    plt.ylabel("Maximum Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Function to run thunderstorm checks
def run_thunderstorm_checks(df, row, thresholds):
    results = {}
    results["Date"] = row["DateT"]
    results["Rain"] = row["Rain"]

    # Initialize all checks as False
    results.update(
        {
            "Humidity_Check": False,
            "Temperature_Check": False,
            "Speed_Check": False,
            "WindDir_Check": False,
            "Pressure_Check": False,
            "Gust_Check": False,
        }
    )

    timestamp = row["DateT"]
    if timestamp - pd.Timedelta(hours=1) in df["DateT"].values:
        prev_row = df.loc[df["DateT"] == timestamp - pd.Timedelta(hours=1)].iloc[0]
        # Iterate over each parameter to check for NaNs and calculate differences
        for param in [
            "Humidity",
            "Temperature",
            "Speed",
            "WindDir",
            "Pressure",
            "Gust",
        ]:
            if not pd.isna(row[param]) and not pd.isna(
                prev_row[param]
            ):  # Check if both current and previous values are not NaN
                diff = row[param] - prev_row[param]
                if param == "Temperature":
                    diff = -diff  # For temperature, a significant drop is a concern
                if param == "WindDir":
                    diff = abs(
                        diff
                    )  # Wind direction check is based on absolute difference
                results[param + "_Check"] = diff > thresholds[param]

    # Determine if there is a thunderstorm based on checks
    results["Thunderstorm"] = (
        int(results["Humidity_Check"])
        + int(results["Temperature_Check"])
        + int(results["Speed_Check"])
        + int(results["WindDir_Check"])
        + int(results["Pressure_Check"])
        + int(results["Gust_Check"])
    ) >= 4

    return results
