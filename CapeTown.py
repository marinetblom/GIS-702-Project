# CT
# only years 1995-98, 2001-2023 should be selected for analysis.
# There are data gaps in 1999 and 2000. Because the data is for Cape Town, the QC procedure should
# in most cases skip step one (test for thunderstorm) and move to step two, which is the test for a
# high rainfall event of synoptic origin. For a start, I propose we test the annual maximum 5-min
# rainfall values,

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno


# Load the file with the correct delimiter and parse the "Date" column as datetime
CT = pd.read_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Data\0021178a3 5min.ttx",
    delimiter="\t",
    encoding="latin1",
    decimal=",",  # Specify comma as decimal separator
    parse_dates=["DateT"],  # Automatically parse 'DateT' as datetime
)

CT.info


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
CT["Pressure"] = CT["Pressure"].apply(clean_pressure)


# Filter the dataframe to exclude 1997 and include only the desired years
filtered_CT = CT[
    (CT["DateT"].dt.year >= 1995) & (CT["DateT"].dt.year <= 1998)
    | (CT["DateT"].dt.year >= 2001) & (CT["DateT"].dt.year <= 2023)
]

# Reset the index to start from 0
filtered_CT.reset_index(drop=True, inplace=True)

# Load in dataframe
filtered_df = filtered_CT.copy()


# Summary statistics function with export to CSV
def summarize_statistics(filtered_df, file_path):
    pd.set_option("display.float_format", lambda x: "%.3f" % x)
    summary = filtered_df.describe()
    print(summary)
    summary.to_csv(file_path)


# Specify the path where you want to save the file
file_path = r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\SummaryStatsCT.csv"

# Summarize statistics and export to CSV
summarize_statistics(filtered_df, file_path)


# Create a visualization dataframe
def prepare_visualization_data(filtered_df):
    # Create a copy for visualizations to ensure the original data stays valid.
    visualization_df = filtered_df.copy()
    return visualization_df


# Prepare data for visualization
visualization_df = prepare_visualization_data(filtered_df)


# Missing Values
def check_missing_data(visualization_df):
    # Identify missing values
    missing_values = visualization_df.isnull().sum()
    print("Missing values per column:\n", missing_values)

    # Calculate the percentage of missing values per column
    missing_percentage = (visualization_df.isnull().sum() / len(visualization_df)) * 100
    print("Percentage of missing values per column:\n", missing_percentage)


# Check for missing data
check_missing_data(visualization_df)

# use missingno library to visualize distribution of missing values
msno.matrix(visualization_df)
msno.dendrogram(visualization_df)


# Function to check for out-of-order timestamps
def check_out_of_order_timestamps(df):
    if not df["DateT"].is_monotonic_increasing:
        print("Data contains out-of-order timestamps.")
    else:
        print("Data contains NO out-of-order timestamps")


# Check for out-of-order timestamps
check_out_of_order_timestamps(visualization_df)


# Function to check for duplicate timestamps
def check_duplicate_timestamps(df):
    duplicate_timestamps = df["DateT"].duplicated().sum()
    if duplicate_timestamps > 0:
        print(f"Data contains {duplicate_timestamps} duplicate timestamps.")
    else:
        print("No duplicate timestamps")


# Check for duplicate timestamps
check_duplicate_timestamps(visualization_df)


# Accuracy of a time measurement
def check_temporal_quality(df):
    # Set DateT as the index in the visualization copy
    df.set_index("DateT", inplace=True)

    # Generate a complete time range with 5-minute intervals
    complete_time_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="5T",
    )

    # Reindex the dataframe to the complete time range to identify missing intervals
    reindexed_df = df.reindex(complete_time_range)

    # Identify the missing timestamps
    missing_timestamps = reindexed_df[reindexed_df.isnull().any(axis=1)].index

    # Display the missing timestamps
    print("Missing Timestamps:\n", missing_timestamps)

    # Aggregate the missing timestamps by year
    missing_by_year = (
        missing_timestamps.to_series()
        .groupby(missing_timestamps.to_series().dt.year)
        .count()
    )

    # Plot the number of missing timestamps per year
    plt.figure(figsize=(12, 6))
    missing_by_year.plot(kind="bar", color="skyblue")
    plt.title("Number of Missing Timestamps per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Missing Timestamps")
    plt.show()


# Check temporal quality
check_temporal_quality(visualization_df)


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

# Save data frame to csv
all_checks_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\StormCT.csv"
)

################################################################

# Frontal rain check
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

# Frontal rain check on the entire annual_max_rainfall DataFrame
frontal_rain_results = []

# Iterate over all rows in the annual_max_rainfall DataFrame
for _, row in annual_max_rainfall.iterrows():
    corresponding_row = filtered_df[filtered_df["DateT"] == row["DateT"]]

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


# Save data frame to csv
frontal_rain_checks_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\FR_CT.csv"
)


################################################################


# Combine thunderstorm and frontal rain check results
final_df = pd.merge(
    all_checks_df[
        ["Date", "Thunderstorm"]
    ],  # Keep only the necessary columns from all_checks_df
    frontal_rain_checks_df[
        ["Date", "FrontalRain"]
    ],  # Keep only the necessary columns from frontal_rain_checks_df
    on="Date",  # Merge on the Date column
)

# Display the final DataFrame showing both checks
print("Final Check Results DataFrame:")
print(final_df)


# PLOT DATA


# Plot all relevant parameters around a given timestamp
def plot_all_parameters_around_event(df, timestamp, window):
    # Define the time range around the timestamp
    time_range = pd.Timedelta(window)
    start_time = timestamp - time_range
    end_time = timestamp + time_range

    # Filter data around the timestamp
    plot_data = df[(df["DateT"] >= start_time) & (df["DateT"] <= end_time)]

    # Create the figure and axes for multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Wind Speed and Gust on the same plot
    axs[0].plot(
        plot_data["DateT"],
        plot_data["Gust"],
        label="Gust (m/s)",
        color="green",
        marker="o",
    )
    axs[0].set_ylabel("Gust (m/s)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title(f"Weather Parameters Around {timestamp}")

    # Temperature on its own plot
    axs[1].plot(
        plot_data["DateT"],
        plot_data["Temperature"],
        label="Temperature (°C)",
        color="red",
        marker="o",
    )
    axs[1].set_ylabel("Temperature (°C)")
    axs[1].legend()
    axs[1].grid(True)

    #  Wind Direction on their own plot
    axs[2].plot(
        plot_data["DateT"],
        plot_data["WindDir"],
        label="WindDir (°)",
        color="cyan",
        marker="o",
    )
    axs[2].set_ylabel("Wind Direction (°)", color="cyan")
    axs[2].legend(loc="upper right")
    axs[2].grid(True)

    # Set x-axis label and format
    axs[2].set_xlabel("Time")
    plt.xticks(rotation=45)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Iterate over rows where FrontalRain is False
for _, row in frontal_rain_checks_df[~frontal_rain_checks_df["FrontalRain"]].iterrows():
    timestamp = row["Date"]
    print(f"Plotting for timestamp: {timestamp}")
    plot_all_parameters_around_event(filtered_df, timestamp, window="60T")


# Manual Verification
def manual_verification(df, specific_timestamp):
    specific_row = df.loc[specific_timestamp]
    preceding_hour_timestamp = specific_timestamp - pd.Timedelta(hours=1)
    preceding_hour_row = df.loc[preceding_hour_timestamp]

    print("Specific Timestamp Row:")
    print(specific_row)

    print("\nPreceding Hour Row:")
    print(preceding_hour_row)


# Manual verification
specific_timestamp = pd.Timestamp("1998-07-06 18:40:00")
manual_verification(filtered_df, specific_timestamp)
