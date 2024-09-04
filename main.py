# main.py
import pandas as pd

from preprocessing import load_cape_town, load_bloemfontein, load_durban

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
    run_thunderstorm_checks,
)

from frontal_rain import apply_frontal_rain_check

# Choose the dataset
dataset_name = input("Enter the dataset name (cape_town, bloemfontein, or durban): ")

if dataset_name == "cape_town":
    filtered_df = load_cape_town()
elif dataset_name == "bloemfontein":
    filtered_df = load_bloemfontein()
elif dataset_name == "durban":
    filtered_df = load_durban()
else:
    raise ValueError(
        "Invalid dataset name provided. Choose 'cape_town', 'bloemfontein', or 'durban'."
    )
# verify if the dataset is loading correctly
print(filtered_df.head())

# verify data types
filtered_df.dtypes

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

# Storm detection process

# Identify annual maximum rainfall
annual_max_rainfall = identify_annual_max_rainfall(filtered_df)
print("Annual Maximum Rainfall Events:\n", annual_max_rainfall)

# Plot the annual maximum rainfall
plot_annual_max_rainfall(annual_max_rainfall)

# Define thresholds for thunderstorm checks
thresholds = {
    "Humidity": 10,  # % increase
    "Temperature": 2,  # °C decrease
    "Speed": 5,  # m/s increase
    "WindDir": 30,  # degrees change
    "Pressure": 1,  # hPa increase
    "Gust": 5,  # m/s increase
}

# Apply the thunderstorm checks
check_results = []
for _, row in annual_max_rainfall.iterrows():
    check_results.append(run_thunderstorm_checks(filtered_df, row, thresholds))

all_checks_df = pd.DataFrame(check_results)

print("Thunderstorm Checks DataFrame:")
print(all_checks_df)

# Save data frame to CSV
all_checks_df.to_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\Results\StormCT.csv"
)

########################################################################

# Apply the frontal rain check to the entire dataset
frontal_rain_checks_df = apply_frontal_rain_check(filtered_df, annual_max_rainfall)

# Print the final results of the frontal rain check
print("Frontal rain Checks DataFrame:")
print(frontal_rain_checks_df)

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
