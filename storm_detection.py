# storm_detection.py
import pandas as pd
import matplotlib.pyplot as plt


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

                # Apply the check and store True/False
                results[param + "_Check"] = diff > thresholds[param]

    # Calculate the weighted score
    weighted_score = sum(
        int(results[param + "_Check"]) * weights[param] for param in weights
    )

    # Define threshold for determining if it's a thunderstorm (adjustable)
    results["Thunderstorm"] = weighted_score >= 4  # You can adjust this threshold

    return results


# Function to apply thunderstorm checks to any dataset
def apply_thunderstorm_checks(df, event_df, thresholds, weights):
    check_results = []
    for _, row in event_df.iterrows():
        check_results.append(run_thunderstorm_checks(df, row, thresholds, weights))

    all_checks_df = pd.DataFrame(check_results)
    return all_checks_df
