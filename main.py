# main.py
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import (
    load_cape_town,
    load_bloemfontein,
    load_durban,
    load_johannesburg,
)

from logical import summarize_statistics

from completeness import check_missing_data

from temporal_quality import (
    check_temporal_quality,
    check_out_of_order_timestamps,
    check_duplicate_timestamps,
)

from rainfall_analysis import (
    identify_annual_max_rainfall,
    plot_annual_max_rainfall,
    apply_pot_filter,
    identify_moving_average_max_rainfall_optimized,
)

from storm_detection import apply_thunderstorm_checks

from frontal_rain import apply_frontal_rain_check

# Choose the dataset
dataset_name = input(
    "Enter the dataset name (0021178a3 (CT), 0261516b0 (Bloem), 0241186a6 (Durban), or 0476399 0 (Jhb)): "
)

if dataset_name == "0021178a3":
    filtered_df = load_cape_town()
elif dataset_name == "0261516b0":
    filtered_df = load_bloemfontein()
elif dataset_name == "0241186a6":
    filtered_df = load_durban()
elif dataset_name == "0476399 0":
    filtered_df = load_johannesburg()
else:
    raise ValueError(
        "Invalid dataset name provided. Choose '0021178a3', '0261516b0', '0241186a6', or '0476399 0'."
    )

# Verify if the dataset is loading correctly
print(filtered_df.head())

# Verify data types
print(filtered_df.dtypes)

################################

# Perform data quality checks

visualization_df = filtered_df.copy()

file_path = f"Results/SummaryStats_{dataset_name}.csv"
summarize_statistics(filtered_df, file_path)

check_missing_data(visualization_df)

check_out_of_order_timestamps(visualization_df)

check_duplicate_timestamps(visualization_df)

check_temporal_quality(visualization_df)

########################################################################

# 1. Identify and plot annual maximum rainfall
annual_max_rainfall = identify_annual_max_rainfall(filtered_df)
print("Annual Maximum Rainfall Events:\n", annual_max_rainfall)
plot_annual_max_rainfall(annual_max_rainfall)


# 2. Define and apply POT filter
threshold_value = float(input("Enter the rainfall threshold value: "))
pot_filtered_df = apply_pot_filter(filtered_df, threshold_value)
print(
    f"Filtered DataFrame for POT with rainfall values greater than {threshold_value}:\n",
    pot_filtered_df,
)
print(f"Number of rows in filtered DataFrame: {pot_filtered_df.shape[0]}")


# 3. Identify the maximum rainfall events for each year based on moving average
window_size_minutes = int(
    input("Enter the window size in minutes (e.g., 60 for 1 hour): ")
)
moving_average_max_rainfall_df = identify_moving_average_max_rainfall_optimized(
    filtered_df, window_size_minutes
)
print(
    "Maximum Rainfall Events Based on Moving Average (Optimized):\n",
    moving_average_max_rainfall_df,
)


################################################################

# Storm detection process

# Define thresholds for thunderstorm checks
thresholds = {
    "Humidity": 10,  # % increase
    "Temperature": 2,  # °C decrease
    "Speed": 5,  # m/s increase
    "WindDir": 30,  # degrees change
    "Pressure": 1,  # hPa increase
    "Gust": 5,  # m/s increase
}

# Define weights for the storm detection checks
storm_weights = {
    "Humidity": 1,
    "Temperature": 1,
    "Speed": 1,
    "WindDir": 1,  #
    "Pressure": 1,  # Lower weight for pressure in storm detection
    "Gust": 2,  # Higher weight for gusts in storm detection
}

# 1. Apply the storm checks to the **annual_max_rainfall** dataset
all_checks_df = apply_thunderstorm_checks(
    filtered_df, annual_max_rainfall, thresholds, storm_weights
)
print("Thunderstorm Checks for Annual Max Rainfall DataFrame:")
print(all_checks_df)

# Save results for annual max rainfall checks
all_checks_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Storm_AnnualMax_Result.csv"
)


# 2. Apply the storm checks to the **POT dataset** ('pot_filtered_df')
all_checks_df_pot = apply_thunderstorm_checks(
    filtered_df, pot_filtered_df, thresholds, storm_weights
)
print("Thunderstorm Checks for POT DataFrame:")
print(all_checks_df_pot)

# Save the POT check results
all_checks_df_pot.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Storm_POT_Result.csv"
)


# 3. Apply the storm checks to the **moving average dataset** ('moving_average_max_rainfall_df')
all_checks_df_ma = apply_thunderstorm_checks(
    filtered_df, moving_average_max_rainfall_df, thresholds, storm_weights
)
print("Thunderstorm Checks for MA DataFrame:")
print(all_checks_df_ma)

# Save the POT check results
all_checks_df_ma.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Storm_MA_Result.csv"
)

########################################################################

# Frontal Rain

# Frontal Rain Check thresholds
frontal_rain_thresholds = {
    "Humidity": 90,  # % absolute threshold or rising to within an hour
    "Humidity_min": 80,  # % minimum
    "Temperature": 1,  # °C decrease
    "WindDir": 30,  # degrees change
    "Pressure": 1,  # Any positive hPa increase
    "Gust": 3,  # m/s decrease
}

# Define weights for frontal rain checks
frontal_rain_weights = {
    "Humidity": 1,
    "Temperature": 1,
    "WindDir": 1,
    "Pressure": 2,  # Higher weight for pressure in frontal rain detection
    "Gust": 1,  # Lower weight for gusts in frontal rain detection
}

# 1. Apply the frontal rain check
frontal_rain_checks_df = apply_frontal_rain_check(
    filtered_df, annual_max_rainfall, frontal_rain_thresholds, frontal_rain_weights
)
print("Frontal Rain Checks DataFrame:")
print(frontal_rain_checks_df)

# Save data frame to CSV
frontal_rain_checks_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Frontal_Rain_AnnualMax_Result.csv"
)


# 2. Apply the frontal rain check to the POT dataset
frontal_rain_checks_df_pot = apply_frontal_rain_check(
    filtered_df, pot_filtered_df, frontal_rain_thresholds, frontal_rain_weights
)
print("Frontal Rain Checks for POT DataFrame:")
print(frontal_rain_checks_df_pot)

# Save the frontal rain check results to CSV
frontal_rain_checks_df_pot.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Frontal_Rain_POT_Result.csv"
)


# 3. Apply the frontal rain check to the **moving average dataset**
frontal_rain_checks_df_ma = apply_frontal_rain_check(
    filtered_df,
    moving_average_max_rainfall_df,
    frontal_rain_thresholds,
    frontal_rain_weights,
)
print("Frontal Rain Checks for Moving Average DataFrame:")
print(frontal_rain_checks_df_ma)

frontal_rain_checks_df_ma.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Frontal_Rain_MA_Result.csv"
)

################################################################

# 1. Combine thunderstorm and frontal rain check results AUNNUAL MAX
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
print("Final Check Results DataFrame for Annual Max:")
print(final_df)

# Save data frame to CSV
final_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Final_ANNUAL.csv"
)

# 2. Combine thunderstorm and frontal rain check results POT
final_df_POT = pd.merge(
    all_checks_df_pot[["Date", "Thunderstorm"]],
    frontal_rain_checks_df_pot[["Date", "FrontalRain"]],
    on="Date",  # Merge on the Date column
)

# Display the final DataFrame showing both checks
print("Final Check Results DataFrame for POT:")
print(final_df_POT)

# Save data frame to CSV
final_df_POT.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Final_POT.csv"
)

# 3. Combine thunderstorm and frontal rain check results MA
final_df_MA = pd.merge(
    all_checks_df_ma[["Date", "Thunderstorm"]],
    frontal_rain_checks_df_ma[["Date", "FrontalRain"]],
    on="Date",  # Merge on the Date column
)

# Display the final DataFrame showing both checks
print("Final Check Results DataFrame for MA:")
print(final_df_MA)

# Save data frame to CSV
final_df_MA.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Final_MA.csv"
)


# Plot all relevant parameters around a given timestamp
def plot_all_parameters_around_event(df, timestamp, window="60T"):
    # Define the time range around the timestamp
    time_range = pd.Timedelta(window)
    start_time = timestamp - time_range
    end_time = timestamp + time_range

    # Filter data around the timestamp
    plot_data = df[(df["DateT"] >= start_time) & (df["DateT"] <= end_time)]

    # Create the figure and axes for multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot Wind Speed and Gust on the same subplot
    axs[0].plot(
        plot_data["DateT"],
        plot_data["Gust"],
        label="Gust (m/s)",
        color="green",
        marker="o",
    )
    axs[0].plot(
        plot_data["DateT"],
        plot_data["Speed"],
        label="Speed (m/s)",
        color="blue",
        marker="o",
    )
    axs[0].set_ylabel("Gust & Speed (m/s)")
    axs[0].legend(loc="upper left")
    axs[0].grid(True)
    axs[0].set_title(f"Weather Parameters Around {timestamp}")

    # Plot Temperature on its own subplot
    axs[1].plot(
        plot_data["DateT"],
        plot_data["Temperature"],
        label="Temperature (°C)",
        color="red",
        marker="o",
    )
    axs[1].set_ylabel("Temperature (°C)")
    axs[1].legend(loc="upper left")
    axs[1].grid(True)

    # Plot Wind Direction on its own subplot
    axs[2].plot(
        plot_data["DateT"],
        plot_data["WindDir"],
        label="Wind Direction (°)",
        color="cyan",
        marker="o",
    )
    axs[2].set_ylabel("Wind Direction (°)")
    axs[2].legend(loc="upper left")
    axs[2].grid(True)

    # Set x-axis label and format
    axs[2].set_xlabel("Time")
    plt.xticks(rotation=45)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


plot_all_parameters_around_event(
    filtered_df, pd.Timestamp("2022-12-12 07:35:00"), window="60T"
)
