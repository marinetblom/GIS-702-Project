# storm_detection.py
import pandas as pd


# Function to run thunderstorm checks
def run_thunderstorm_checks(df, row, thresholds, weights):
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
            "Rain_Check": False,  # Add a new Rain Check
        }
    )

    timestamp = row["DateT"]

    # Check for previous and next 5-minute interval rain data
    prev_5min = timestamp - pd.Timedelta(minutes=5)
    next_5min = timestamp + pd.Timedelta(minutes=5)

    # Get the previous and next rows if available
    prev_rain_row = df.loc[df["DateT"] == prev_5min]
    next_rain_row = df.loc[df["DateT"] == next_5min]

    if not prev_rain_row.empty and not next_rain_row.empty:
        prev_rain = prev_rain_row.iloc[0]["Rain"]
        next_rain = next_rain_row.iloc[0]["Rain"]

        # Check if rain in both intervals is greater than 0mm
        results["Rain_Check"] = prev_rain > 0 or next_rain > 0

    # Existing checks for the last 1 hour
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

                # Apply the check and store True/False
                results[param + "_Check"] = diff > thresholds[param]

    # Calculate the weighted score including the new Rain_Check
    weighted_score = sum(
        int(results[param + "_Check"]) * weights[param] for param in weights
    )

    # Define threshold for determining if it's a thunderstorm (adjustable)
    results["Thunderstorm"] = weighted_score >= 5  # You can adjust this threshold

    return results


# Function to apply thunderstorm checks to any dataset
def apply_thunderstorm_checks(df, event_df, thresholds, weights):
    check_results = []
    for _, row in event_df.iterrows():
        check_results.append(run_thunderstorm_checks(df, row, thresholds, weights))

    all_checks_df = pd.DataFrame(check_results)
    return all_checks_df
