# frontal_rain.py
import pandas as pd


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
def run_frontal_rain_checks(df, row, thresholds, weights):
    results = {}
    results["Date"] = row["DateT"]
    results["Rain"] = row["Rain"]

    timestamp = row["DateT"]
    six_hours_after = timestamp + pd.Timedelta(hours=6)

    # Initialize all checks as False
    results.update(
        {
            "F_Temperature_Check": False,
            "F_Humidity_Check": False,
            "F_WindDir_Check": False,
            "F_Pressure_Check": False,
            "F_Rain_Check": False,  # New Rain Check
        }
    )

    # Calculate the averages before and after the event for Temperature
    before_temp_avg = calculate_average_in_window(
        df, timestamp, "before", param="Temperature"
    )
    after_temp_avg = calculate_average_in_window(
        df, timestamp, "after", param="Temperature"
    )

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

    # New Rain Check: Check for rainfall 5 minutes before and after the event
    prev_5min = timestamp - pd.Timedelta(minutes=5)
    next_5min = timestamp + pd.Timedelta(minutes=5)

    # Get the previous and next rows if available
    prev_rain_row = df[df["DateT"] == prev_5min]
    next_rain_row = df[df["DateT"] == next_5min]

    if not prev_rain_row.empty and not next_rain_row.empty:
        prev_rain = prev_rain_row.iloc[0]["Rain"]
        next_rain = next_rain_row.iloc[0]["Rain"]

        # Check if rain in either interval is greater than 1mm
        if prev_rain > 0 and next_rain > 0:
            results["F_Rain_Check"] = True

    # Calculate the weighted score based on the weights provided
    weighted_score = (
        int(results["F_Temperature_Check"]) * weights["Temperature"]
        + int(results["F_Humidity_Check"]) * weights["Humidity"]
        + int(results["F_WindDir_Check"]) * weights["WindDir"]
        + int(results["F_Pressure_Check"]) * weights["Pressure"]
        + int(results["F_Rain_Check"]) * weights["Rain"]  # New Rain check score
    )

    # Determine if it meets frontal rain criteria based on the weighted score
    results["FrontalRain"] = weighted_score >= 4  # You can adjust this threshold

    return results


# Function to run frontal rain check on the entire DataFrame
def apply_frontal_rain_check(df, event_df, thresholds, weights):
    frontal_rain_results = []
    for _, row in event_df.iterrows():
        corresponding_row = df[df["DateT"] == row["DateT"]]
        if not corresponding_row.empty:
            corresponding_row = corresponding_row.iloc[0]
            # Perform the frontal rain check
            frontal_rain_results.append(
                run_frontal_rain_checks(df, corresponding_row, thresholds, weights)
            )
    # Convert to DataFrame
    frontal_rain_checks_df = pd.DataFrame(frontal_rain_results)
    return frontal_rain_checks_df
