# The data to be considered is for the periods 1995-1996, 1998-2023. The 1997 data has some gaps in
# February, and late-summer is usually the period that the western parts of the summer rainfall region
# receive most of their rainfall – so it is best to ignore this year.
# For this station the high rainfall values will probably be all due to thunderstorms. If you haven’t
# started with the programming, it is probably best to start with this weather station – verification of
# thunderstorms is arguably the most critical component of the research.

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: LOAD AND INSPECT THE DATA
# Load the file with the correct delimiter and parse the "Date" column as datetime
bloem = pd.read_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Data\0261516b0 5min.ttx",
    delimiter="\t",
    encoding="latin1",
    decimal=",",  # Specify comma as decimal separator
    parse_dates=["DateT"],  # Automatically parse 'DateT' as datetime
)

# Filter the dataframe to exclude 1997 and include only the desired years: Instructions from supervisor
filtered_bloem = bloem[
    (bloem["DateT"].dt.year >= 1995) & (bloem["DateT"].dt.year <= 1996)
    | (bloem["DateT"].dt.year >= 1998) & (bloem["DateT"].dt.year <= 2023)
]

# Reset the index to start from 0
filtered_bloem.reset_index(drop=True, inplace=True)

# Display the first few rows and data types
print(filtered_bloem.head())

################################################################

filtered_df = filtered_bloem.copy()


# Summary statistics function
def summarize_statistics(filtered_df):
    pd.set_option("display.float_format", lambda x: "%.3f" % x)
    print(filtered_df.describe())


# Summarize statistics
summarize_statistics(filtered_df)


def prepare_visualization_data(filtered_df):
    # Create a copy for visualizations to ensure the original data stays valid.
    visualization_df = filtered_df.copy()
    # Extract the date (year, month, day) from the DateT column in the visualization copy
    visualization_df["Date"] = visualization_df["DateT"].dt.date
    return visualization_df


# Prepare data for visualization
visualization_df = prepare_visualization_data(filtered_df)


def check_missing_data(visualization_df):
    # Identify missing values
    missing_values = visualization_df.isnull().sum()
    print("Missing values per column:\n", missing_values)

    # Calculate the percentage of missing values per column
    missing_percentage = (visualization_df.isnull().sum() / len(visualization_df)) * 100
    print("Percentage of missing values per column:\n", missing_percentage)


# Check for missing data
check_missing_data(visualization_df)


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


# Function to check temporal quality
def check_temporal_quality(df):
    # 2.3.1 Temporal consistency
    if not df.index.is_monotonic_increasing:
        print("Timestamps are not in chronological order.")
    else:
        print("Timestamps are in chronological order.")

    # 2.3.2 Accuracy of a time measurement
    # MISSING TIME STAMPS
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


# Thunderstorm detection process
# STEP 1: IDENTIFY ANNUAL MAX RAINFALL
print(filtered_df.head())


# Function to identify annual maximum rainfall
def identify_annual_max_rainfall(df):
    annual_max_rainfall = df.loc[df.groupby(df["DateT"].dt.year)["Rain"].idxmax()]
    print("Annual Maximum Rainfall Events:\n", annual_max_rainfall)


identify_annual_max_rainfall(filtered_df)

# Identify annual maximum rainfall
identify_annual_max_rainfall(filtered_df)
annual_max_rainfall = filtered_df.loc[
    filtered_df.groupby(filtered_df["DateT"].dt.year)["Rain"].idxmax()
]


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

# SPIKE DETECTION

filtered_df["DateT"] = pd.to_datetime(filtered_df["DateT"])
filtered_df.set_index("DateT", inplace=True)


# Function to detect spikes
def detect_spikes(df, time_index, pre_window="15min", spike_threshold=5):
    rolling_max_pre = (
        df[["Speed", "Gust"]].rolling(window=pre_window, closed="left").max()
    )
    pre_max_values = rolling_max_pre.loc[time_index]

    one_hour_before = df.loc[time_index - pd.Timedelta(hours=1)]

    print(f"At {time_index} - 15min window max: {pre_max_values}")
    print(
        f"At {time_index - pd.Timedelta(hours=1)} - One hour before values: {one_hour_before}"
    )

    speed_spike = (pre_max_values["Speed"] - one_hour_before["Speed"]) > spike_threshold
    gust_spike = (pre_max_values["Gust"] - one_hour_before["Gust"]) > spike_threshold

    return speed_spike, gust_spike


# Detect spikes and update checks
for index, row in all_checks_df.iterrows():
    if not row["Thunderstorm"]:
        time_index = row["Date"]
        speed_spike, gust_spike = detect_spikes(filtered_df, time_index)
        if speed_spike:
            all_checks_df.at[index, "Speed_Check"] = True
        if gust_spike:
            all_checks_df.at[index, "Gust_Check"] = True
        all_checks_df.at[index, "Thunderstorm"] = (
            int(all_checks_df.at[index, "Humidity_Check"])
            + int(all_checks_df.at[index, "Temperature_Check"])
            + int(all_checks_df.at[index, "Speed_Check"])
            + int(all_checks_df.at[index, "WindDir_Check"])
            + int(all_checks_df.at[index, "Pressure_Check"])
            + int(all_checks_df.at[index, "Gust_Check"])
        ) >= 4

print("Updated Thunderstorm Checks DataFrame:")
print(all_checks_df)


# PLOT DATA


# Plot all relevant parameters around a given timestamp
def plot_all_parameters_around_event(df, timestamp, window):
    # Define the time range around the timestamp
    time_range = pd.Timedelta(window)
    start_time = timestamp - time_range
    end_time = timestamp + time_range

    # Filter data around the timestamp
    plot_data = df.loc[
        start_time:end_time, ["Speed", "Gust", "Temperature", "Humidity"]
    ]

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Define colors for each parameter
    colors = {
        "Speed": "blue",
        "Gust": "green",
        "Temperature": "red",
        "Humidity": "orange",
    }

    # Plot each parameter with a different color
    for param, color in colors.items():
        plt.plot(
            plot_data.index,
            plot_data[param],
            label=f"{param} ({'m/s' if param in ['Speed', 'Gust'] else '°C' if param == 'Temperature' else '%' if param == 'Humidity' else 'hPa' if param == 'Pressure' else '°'})",
            color=color,
            marker="o",
            linestyle="-",
        )

    plt.title(f"Weather Parameters Around {timestamp}")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage for plotting all parameters
plot_all_parameters_around_event(
    filtered_df, pd.Timestamp("2012-12-25 09:25:00"), window="60T"
)


def calculate_differences(df, annual_max_rainfall):
    difference_results = []
    for _, row in annual_max_rainfall.iterrows():
        timestamp = row["DateT"]
        one_hour_back = timestamp - pd.Timedelta(hours=1)
        if one_hour_back in df.index:
            prev_row = df.loc[one_hour_back]
            differences = {
                "humidity_diff": row["Humidity"] - prev_row["Humidity"],
                "temperature_diff": row["Temperature"] - prev_row["Temperature"],
                "speed_diff": row["Speed"] - prev_row["Speed"],
                "winddir_diff": abs(row["WindDir"] - prev_row["WindDir"]),
                "pressure_diff": row["Pressure"] - prev_row["Pressure"],
                "gust_diff": row["Gust"] - prev_row["Gust"],
            }
            difference_results.append(differences)
    return pd.DataFrame(difference_results)


# Calculate the differences
differences_df = calculate_differences(filtered_df, annual_max_rainfall)

# Display summary statistics of the differences
summary_statistics = differences_df.describe()
print(summary_statistics)


# Function for manual verification
def manual_verification(df, specific_timestamp):
    specific_row = df.loc[specific_timestamp]
    preceding_hour_timestamp = specific_timestamp - pd.Timedelta(hours=1)
    preceding_hour_row = df.loc[preceding_hour_timestamp]

    print("Specific Timestamp Row:")
    print(specific_row)

    print("\nPreceding Hour Row:")
    print(preceding_hour_row)


# Manual verification
specific_timestamp = pd.Timestamp("2005-01-17 22:40:00")
manual_verification(filtered_df, specific_timestamp)
