# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Use 'date_format' to directly specify the date format for parsing
durban = pd.read_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Data\0241186a6 5min.ttx",
    delimiter="\t",
    encoding="latin1",
    decimal=",",  # Specify comma as decimal separator
    parse_dates=["DateT"],  # Automatically parse 'DateT' as datetime
    date_format="%m/%d/%Y %I:%M:%S%p",  # Use 'date_format' instead of 'date_parser'
)


# STEP 2: DEFINE THE CLEANING FUNCTION
def clean_pressure(value):
    if isinstance(value, str):
        cleaned_value = value.replace("\xa0", "").replace(" ", "").replace(",", ".")
        try:
            return float(cleaned_value)
        except ValueError:
            return np.nan
    else:
        return np.nan


# STEP 3: APPLY THE CLEANING FUNCTION
durban["Pressure"] = durban["Pressure"].apply(clean_pressure)

# STEP 4: VERIFY THE RESULTS
print(durban["Pressure"].head(10))
print(durban["Pressure"].dtypes)  # Should output: float64

# Load in dataframe
filtered_df = durban.copy()


filtered_df.dtypes

filtered_df.describe()


# Storm detection process
# STEP 1: IDENTIFY ANNUAL MAX RAINFALL
print(filtered_df.head())


# Function to identify annual maximum rainfall
def identify_annual_max_rainfall(df):
    # Function to identify annual maximum rainfall
    return df.loc[df.groupby(df["DateT"].dt.year)["Rain"].idxmax()]


# Assign annual maximum rainfall using the function
annual_max_rainfall = identify_annual_max_rainfall(filtered_df)
print("Annual Maximum Rainfall Events:\n", annual_max_rainfall)


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


# Plot the annual maximum rainfall
plot_annual_max_rainfall(annual_max_rainfall)


# Define thresholds for thunderstorm checks with lowercase keys
thresholds = {
    "Humidity": 10,  # % increase
    "Temperature": 2,  # °C decrease
    "Speed": 5,  # m/s increase
    "WindDir": 30,  # degrees change
    "Pressure": 1,  # hPa increase
    "Gust": 5,  # m/s increase
}


################################################################
# Define a function to perform thunderstorm checks
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


# Apply the thunderstorm checks
check_results = []
for _, row in annual_max_rainfall.iterrows():
    check_results.append(run_thunderstorm_checks(filtered_df, row, thresholds))

all_checks_df = pd.DataFrame(check_results)

print("Thunderstorm Checks DataFrame:")
print(all_checks_df)


################################################################

# Frontal rain check
# Define thresholds for frontal rain checks with lowercase keys
frontal_rain_thresholds = {
    "Humidity": 90,  # % absolute threshold or rising to within an hour
    "Humidity_min": 80,  # % minimum
    "Temperature": 1,  # °C decrease
    "Speed": 2,  # m/s decrease
    "WindDir": 30,  # degrees change
    "Pressure": 1,  # Any positive hPa increase
    "Gust": 3,  # m/s decrease
}


# Function to calculate the average over a 1-hour window (12 timestamps)
def calculate_average_in_window(df, timestamp, direction, param):
    if direction == "before":
        window_df = df[
            (df["DateT"] <= timestamp)
            & (df["DateT"] > timestamp - pd.Timedelta(hours=1))
        ]
    elif direction == "after":
        window_df = df[
            (df["DateT"] >= timestamp)
            & (df["DateT"] < timestamp + pd.Timedelta(hours=1))
        ]

    # Calculate the mean for the specified parameter, ignoring NaN values
    param_avg = window_df[param].mean()

    return param_avg


# Define the frontal rain check function
def run_frontal_rain_checks(df, row, thresholds):
    results = {}
    results["Date"] = row["DateT"]  # Using DateT from filtered_df
    results["Rain"] = row["Rain"]

    timestamp = row["DateT"]
    six_hours_after = timestamp + pd.Timedelta(hours=6)

    # Initialize all checks as False
    results.update(
        {
            "F_Gust_Check": False,
            "F_Temperature_Check": False,
            "F_Humidity_Check": False,
            "F_WindDir_Check": False,
            "F_Pressure_Check": False,
        }
    )

    # Calculate the averages before and after the event for Gust and Temperature
    before_gust_avg = calculate_average_in_window(df, timestamp, "before", param="Gust")
    after_gust_avg = calculate_average_in_window(df, timestamp, "after", param="Gust")

    before_temp_avg = calculate_average_in_window(
        df, timestamp, "before", param="Temperature"
    )
    after_temp_avg = calculate_average_in_window(
        df, timestamp, "after", param="Temperature"
    )

    # Debugging: Print the averages
    print(f"Timestamp: {timestamp}")
    print(f"Before Gust Avg: {before_gust_avg}, After Gust Avg: {after_gust_avg}")
    print(f"Before Temp Avg: {before_temp_avg}, After Temp Avg: {after_temp_avg}")

    # Wind Gust Decrease: Compare the average one hour before the event with one hour after the event
    gust_diff = before_gust_avg - after_gust_avg
    if gust_diff >= thresholds["Gust"]:
        results["F_Gust_Check"] = True
    print(
        f"Gust Diff: {gust_diff}, Threshold: {thresholds['Gust']}, Check: {results['F_Gust_Check']}"
    )

    # Temperature Decrease: Check if there is any decrease in temperature
    temp_diff = before_temp_avg - after_temp_avg
    if temp_diff > 0:  # Check if temperature decreased
        results["F_Temperature_Check"] = True
    print(f"Temp Diff: {temp_diff}, Check: {results['F_Temperature_Check']}")

    # Humidity Check
    if row["Humidity"] >= thresholds["Humidity"]:
        results["F_Humidity_Check"] = True
    else:
        # Check if humidity rises from above 80% to at least 90% within an hour after the event
        one_hour_row = df[(df["DateT"] == timestamp + pd.Timedelta(hours=1))]
        if not one_hour_row.empty:
            one_hour_row = one_hour_row.iloc[0]
            if (
                row["Humidity"] > thresholds["Humidity_min"]
                and one_hour_row["Humidity"] >= thresholds["Humidity"]
            ):
                results["F_Humidity_Check"] = True

    # Normalize wind direction for wrap-around (add 360 to directions < 90)
    def normalize_wind_dir(wind_dir):
        return wind_dir + 360 if wind_dir < 90 else wind_dir

    # Wind Direction Change: Compare 1 hour before the event with the event timestamp
    before_temp_row = df[(df["DateT"] == timestamp - pd.Timedelta(hours=1))]
    if not before_temp_row.empty:
        before_temp_row = before_temp_row.iloc[0]
        initial_wind_dir = normalize_wind_dir(before_temp_row["WindDir"])
        event_wind_dir = normalize_wind_dir(row["WindDir"])
        if (initial_wind_dir - event_wind_dir) >= thresholds["WindDir"]:
            results["F_WindDir_Check"] = True

    # Surface Pressure Increase: Compare event timestamp with 6 hours later
    six_hours_row = df[(df["DateT"] == six_hours_after)]
    if not six_hours_row.empty:
        six_hours_row = six_hours_row.iloc[0]
        if six_hours_row["Pressure"] > row["Pressure"]:
            results["F_Pressure_Check"] = True

    # Determine if it meets frontal rain criteria
    results["FrontalRain"] = (
        int(results["F_Gust_Check"])
        + int(results["F_Temperature_Check"])
        + int(results["F_Humidity_Check"])
        + int(results["F_WindDir_Check"])
        + int(results["F_Pressure_Check"])
    ) >= 3

    return results


# Application of the Function
frontal_rain_results = []

# Iterate over rows where Thunderstorm is False
for _, row in all_checks_df[all_checks_df["Thunderstorm"] == False].iterrows():
    corresponding_row = filtered_df[filtered_df["DateT"] == row["Date"]]

    # Only proceed if a corresponding row is found
    if not corresponding_row.empty:
        corresponding_row = corresponding_row.iloc[0]

        # Perform the frontal rain check using data from filtered_df
        frontal_rain_results.append(
            run_frontal_rain_checks(
                filtered_df, corresponding_row, frontal_rain_thresholds
            )
        )

# Convert the list of results into a DataFrame
frontal_rain_checks_df = pd.DataFrame(frontal_rain_results)

print("Frontal rain Checks DataFrame:")
print(frontal_rain_checks_df)
