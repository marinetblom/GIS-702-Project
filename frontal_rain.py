# frontal_rain.py
import pandas as pd

# Define thresholds for frontal rain checks with lowercase keys
frontal_rain_thresholds = {
    "Humidity": 90,  # % absolute threshold or rising to within an hour
    "Humidity_min": 80,  # % minimum
    "Temperature": 1,  # °C decrease
    "Speed": 3,  # m/s decrease
    "WindDir": 30,  # degrees change
    "Pressure": 1,  # Any positive hPa increase
    "Gust": 3,  # m/s decrease
}


# Define the frontal rain check function
def run_frontal_rain_checks(df, row, thresholds):
    results = {}
    results["Date"] = row["DateT"]  # Using DateT from filtered_df
    results["Rain"] = row["Rain"]

    timestamp = row["DateT"]
    one_hour_before = timestamp - pd.Timedelta(hours=1)
    one_hour_after = timestamp + pd.Timedelta(hours=1)
    six_hours_after = timestamp + pd.Timedelta(hours=6)

    # Initialize all checks as False
    results.update(
        {
            "F_WindSpeed_Check": False,
            "F_Gust_Check": False,
            "F_Temperature_Check": False,
            "F_Humidity_Check": False,
            "F_WindDir_Check": False,
            "F_Pressure_Check": False,
        }
    )

    # Fetch corresponding rows from the filtered DataFrame
    before_row = df[df["DateT"] == one_hour_before]
    after_row = df[df["DateT"] == one_hour_after]
    six_hours_row = df[df["DateT"] == six_hours_after]

    # Initialize one_hour_row in case it's not defined later
    one_hour_row = pd.DataFrame()

    if not before_row.empty and not after_row.empty:
        before_row = before_row.iloc[0]
        after_row = after_row.iloc[0]

        # Wind Speed Decrease: Compare one hour before the event with one hour after the event
        if (before_row["Speed"] - after_row["Speed"]) >= thresholds["Speed"]:
            results["F_WindSpeed_Check"] = True

        # Wind Gust Decrease: Compare one hour before the event with one hour after the event
        if (before_row["Gust"] - after_row["Gust"]) >= thresholds["Gust"]:
            results["F_Gust_Check"] = True

        # Temperature Decrease: Compare one hour before the event with the event timestamp
        if (before_row["Temperature"] - row["Temperature"]) >= thresholds[
            "Temperature"
        ]:
            results["F_Temperature_Check"] = True

        # Humidity Check
        if row["Humidity"] >= thresholds["Humidity"]:
            results["F_Humidity_Check"] = True
        else:
            # Check if humidity rises from above 80% to at least 90% within an hour after the event
            one_hour_after = timestamp + pd.Timedelta(hours=1)
            one_hour_row = df[df["DateT"] == one_hour_after]

        if not one_hour_row.empty:
            one_hour_row = one_hour_row.iloc[0]
            if (
                row["Humidity"] > thresholds["Humidity_min"]
                and one_hour_row["Humidity"] >= thresholds["Humidity"]
            ):
                results["F_Humidity_Check"] = True

        # Normalize wind direction for wrap-around (add 360 to directions < 90)
        def normalize_wind_dir(wind_dir):
            if wind_dir < 90:
                return wind_dir + 360
            return wind_dir

        # Wind Direction Change: Compare 1 hour before the event with the event timestamp
        initial_wind_dir = normalize_wind_dir(before_row["WindDir"])
        event_wind_dir = normalize_wind_dir(row["WindDir"])

        if (initial_wind_dir - event_wind_dir) >= thresholds["WindDir"]:
            results["F_WindDir_Check"] = True

        # Surface Pressure Increase
        if not six_hours_row.empty:
            six_hours_row = six_hours_row.iloc[0]
            if six_hours_row["Pressure"] > row["Pressure"]:
                results["F_Pressure_Check"] = True

    # Determine if it meets frontal rain criteria
    results["FrontalRain"] = (
        int(results["F_WindSpeed_Check"])
        + int(results["F_Gust_Check"])
        + int(results["F_Temperature_Check"])
        + int(results["F_Humidity_Check"])
        + int(results["F_WindDir_Check"])
        + int(results["F_Pressure_Check"])
    ) >= 4

    return results


# Function to apply the frontal rain check
def apply_frontal_rain_checks(filtered_df, all_checks_df):
    frontal_rain_results = []

    # Iterate over rows where Thunderstorm is False
    for _, row in all_checks_df[~all_checks_df["Thunderstorm"]].iterrows():
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

    # Save data frame to CSV
    frontal_rain_checks_df.to_csv(
        r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\FR_CT.csv"
    )
