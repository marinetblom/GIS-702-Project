# CT
# only years 1995-98, 2001-2023 should be selected for analysis.
# There are data gaps in 1999 and 2000. Because the data is for Cape Town, the QC procedure should
# in most cases skip step one (test for thunderstorm) and move to step two, which is the test for a
# high rainfall event of synoptic origin. For a start, I propose we test the annual maximum 5-min
# rainfall values,

# STEP 1: LOAD AND INSPECT THE DATA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the file with the correct delimiter and parse the "Date" column as datetime
CT = pd.read_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Data\0021178a3 5min.ttx",
    delimiter="\t",
    encoding="latin1",
    decimal=",",  # Specify comma as decimal separator
    parse_dates=["DateT"],  # Automatically parse 'DateT' as datetime
)


def clean_pressure(value):
    if isinstance(value, str):
        # Remove non-numeric and non-decimal characters, specifically targeting escaped characters
        cleaned_value = "".join(c for c in value if c.isdigit() or c == "." or c == ",")
        # Replace comma with dot in case it's used as decimal separator
        cleaned_value = cleaned_value.replace(",", ".")
        try:
            # Try to convert the cleaned string to float
            return float(cleaned_value)
        except ValueError:
            # Return None or np.nan to identify values that still cause issues
            return np.nan


# Apply the cleaning function to the Pressure column
CT["Pressure"] = CT["Pressure"].apply(clean_pressure)

# Explicitly change the dtype to float64 after cleaning
CT["Pressure"] = CT["Pressure"].astype("float64")

# Check if the pressure column is correct
print(CT.head())
print(CT.dtypes)

# Filter the dataframe to exclude 1997 and include only the desired years
filtered_CT = CT[
    (CT["DateT"].dt.year >= 1995) & (CT["DateT"].dt.year <= 1998)
    | (CT["DateT"].dt.year >= 2001) & (CT["DateT"].dt.year <= 2023)
]

# Reset the index to start from 0
filtered_CT.reset_index(drop=True, inplace=True)
print(filtered_CT.head(10))

# Load in dataframe

filtered_df = filtered_CT.copy()


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


# Storm detection process
# STEP 1: IDENTIFY ANNUAL MAX RAINFALL
print(filtered_df.head())


# Function to identify annual maximum rainfall
def identify_annual_max_rainfall(df):
    annual_max_rainfall = df.loc[df.groupby(df["DateT"].dt.year)["Rain"].idxmax()]
    print("Annual Maximum Rainfall Events:\n", annual_max_rainfall)


# Identify annual maximum rainfall
annual_max_rainfall = identify_annual_max_rainfall(filtered_df)
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

print("Storm Checks DataFrame:")
print(all_checks_df)

################################################################


# Define thresholds for frontal rain checks with lowercase keys
frontal_rain_thresholds = {
    "Humidity": 90,  # % absolute threshold or rising to within an hour
    "Humidity_min": 80,  # % minimum
    "Temperature": 1,  # °C decrease
    "Speed": 1,  # m/s decrease
    "WindDir": 30,  # degrees change
    "Pressure": 1,  # Any positive hPa increase
    "Gust": 1,  # m/s decrease
}


# Define the frontal rain check function
def run_frontal_rain_checks(df, row, thresholds):
    results = {}
    results["Date"] = row["DateT"]  # Using DateT from filtered_df
    results["Rain"] = row["Rain"]

    timestamp = row["DateT"]
    thirty_min_before = timestamp - pd.Timedelta(minutes=30)
    thirty_min_after = timestamp + pd.Timedelta(minutes=30)

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
    before_row = df[df["DateT"] == thirty_min_before]
    after_row = df[df["DateT"] == thirty_min_after]

    six_hours_row = df[df["DateT"] == six_hours_after]

    # Initialize one_hour_row in case it's not defined later
    one_hour_row = pd.DataFrame()

    if not before_row.empty and not after_row.empty:
        before_row = before_row.iloc[0]
        after_row = after_row.iloc[0]

        # Wind Speed and Gust Decrease 30 minutes after the event
        if (row["Speed"] - after_row["Speed"]) >= thresholds["Speed"]:
            results["F_WindSpeed_Check"] = True

        if (row["Gust"] - after_row["Gust"]) >= thresholds["Gust"]:
            results["F_Gust_Check"] = True

        # Temperature Decrease: Compare 30 minutes before the event with 30 minutes after the event
        if (before_row["Temperature"] - after_row["Temperature"]) >= thresholds[
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

        # Wind Direction Change: Check for a decrease within a broader time window
        time_points = [
            pd.Timedelta(minutes=10),
            pd.Timedelta(minutes=20),
            pd.Timedelta(minutes=30),
        ]
        initial_wind_dir = normalize_wind_dir(row["WindDir"])
        wind_dir_decrease = False

        for time_point in time_points:
            check_time = timestamp + time_point
            check_row = df[df["DateT"] == check_time]

            if not check_row.empty:
                check_wind_dir = normalize_wind_dir(check_row.iloc[0]["WindDir"])
                if initial_wind_dir > check_wind_dir:
                    wind_dir_decrease = True
                    break  # Exit loop as soon as a decrease is found

        # Mark the check as passed if a decrease was detected
        if wind_dir_decrease:
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


################################################################


filtered_df["DateT"] = pd.to_datetime(filtered_df["DateT"])
filtered_df.set_index("DateT", inplace=True)


# PLOT DATA


# Plot all relevant parameters around a given timestamp
def plot_all_parameters_around_event(df, timestamp, window):
    # Define the time range around the timestamp
    time_range = pd.Timedelta(window)
    start_time = timestamp - time_range
    end_time = timestamp + time_range

    # Filter data around the timestamp
    plot_data = df.loc[start_time:end_time, ["Speed", "Gust", "Temperature"]]

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Define colors for each parameter
    colors = {
        "Speed": "blue",
        "Gust": "green",
        "Temperature": "red",
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
    filtered_df, pd.Timestamp("2023-08-25 11:25:00"), window="60T"
)


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
