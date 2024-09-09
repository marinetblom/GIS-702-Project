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

from storm_detection import (
    identify_annual_max_rainfall,
    plot_annual_max_rainfall,
    apply_thunderstorm_checks,
)

from frontal_rain import apply_frontal_rain_check

# Choose the dataset
dataset_name = input(
    "Enter the dataset name (cape_town, bloemfontein, durban, or johannesburg): "
)

if dataset_name == "cape_town":
    filtered_df = load_cape_town()
elif dataset_name == "bloemfontein":
    filtered_df = load_bloemfontein()
elif dataset_name == "durban":
    filtered_df = load_durban()
elif dataset_name == "johannesburg":
    filtered_df = load_johannesburg()
else:
    raise ValueError(
        "Invalid dataset name provided. Choose 'cape_town', 'bloemfontein', 'durban', or 'johannesburg'."
    )

# Verify if the dataset is loading correctly
print(filtered_df.head())

# Verify data types
print(filtered_df.dtypes)

################################

# Perform data quality checks
visualization_df = filtered_df.copy()

# Perform summary statistics and export to CSV
file_path = f"Results/SummaryStats_{dataset_name}.csv"
summarize_statistics(filtered_df, file_path)

check_missing_data(visualization_df)

check_out_of_order_timestamps(visualization_df)

check_duplicate_timestamps(visualization_df)

check_temporal_quality(visualization_df)

########################################################################

# Identify annual maximum rainfall
annual_max_rainfall = identify_annual_max_rainfall(filtered_df)
print("Annual Maximum Rainfall Events:\n", annual_max_rainfall)
plot_annual_max_rainfall(annual_max_rainfall)

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

# Apply the storm checks to the dataset
all_checks_df = apply_thunderstorm_checks(filtered_df, annual_max_rainfall, thresholds)
print("Thunderstorm Checks DataFrame:")
print(all_checks_df)

# Save data frame to CSV
all_checks_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Storm_Result.csv"
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

# Apply the frontal rain check
frontal_rain_checks_df = apply_frontal_rain_check(
    filtered_df, annual_max_rainfall, frontal_rain_thresholds
)
print("Frontal Rain Checks DataFrame:")
print(frontal_rain_checks_df)

# Save data frame to CSV
frontal_rain_checks_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Frontal_Rain_Result.csv"
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

# Save data frame to CSV
final_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\Final.csv"
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
