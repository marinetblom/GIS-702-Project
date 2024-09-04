# frontal_rain.py
import pandas as pd

# Define thresholds for frontal rain checks with lowercase keys
frontal_rain_thresholds = {
    "Humidity": 90,  # % absolute threshold or rising to within an hour
    "Humidity_min": 80,  # % minimum
    "Temperature": 1,  # °C decrease
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
def run_frontal_rain_checks(df, row, thresholds=frontal_rain_thresholds):
    results = {}
    results["Date"] = row["DateT"]
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

    # Wind Gust Decrease: Compare the average one hour before the event with one hour after the event
    gust_diff = before_gust_avg - after_gust_avg
    if gust_diff >= thresholds["Gust"]:
        results["F_Gust_Check"] = True

    # Temperature Decrease: Check if there is any decrease in temperature
    temp_diff = before_temp_avg - after_temp_avg
    if temp_diff > 0:
        results["F_Temperature_Check"] = True

    # Humidity Check
    if row["Humidity"] >= thresholds["Humidity"]:
        results["F_Humidity_Check"] = True
    else:
        # Check if humidity rises from above 80% to at least 90% within an hour after the event
        one_hour_row = df[df["DateT"] == timestamp + pd.Timedelta(hours=1)]
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
    before_temp_row = df[df["DateT"] == timestamp - pd.Timedelta(hours=1)]
    if not before_temp_row.empty:
        before_temp_row = before_temp_row.iloc[0]
        initial_wind_dir = normalize_wind_dir(before_temp_row["WindDir"])
        event_wind_dir = normalize_wind_dir(row["WindDir"])
        if (initial_wind_dir - event_wind_dir) >= thresholds["WindDir"]:
            results["F_WindDir_Check"] = True

    # Surface Pressure Increase: Compare event timestamp with 6 hours later
    six_hours_row = df[df["DateT"] == six_hours_after]
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


# Function to run frontal rain check on the entire DataFrame
def apply_frontal_rain_check(df, annual_max_rainfall):
    frontal_rain_results = []
    for _, row in annual_max_rainfall.iterrows():
        corresponding_row = df[df["DateT"] == row["DateT"]]
        if not corresponding_row.empty:
            corresponding_row = corresponding_row.iloc[0]
            # Perform the frontal rain check
            frontal_rain_results.append(
                run_frontal_rain_checks(df, corresponding_row, frontal_rain_thresholds)
            )
    # Convert to DataFrame
    frontal_rain_checks_df = pd.DataFrame(frontal_rain_results)
    return frontal_rain_checks_df
