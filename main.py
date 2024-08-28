# main.py
import pandas as pd

from preprocessing import load_cape_town, load_bloemfontein

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

from frontal_rain import apply_frontal_rain_checks

# Choose the dataset
dataset_name = input("Enter the dataset name (cape_town or bloemfontein): ")

if dataset_name == "cape_town":
    filtered_df = load_cape_town()
elif dataset_name == "bloemfontein":
    filtered_df = load_bloemfontein()
else:
    raise ValueError(
        "Invalid dataset name provided. Choose 'cape_town' or 'bloemfontein'."
    )

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

# Apply frontal rain checks
apply_frontal_rain_checks(filtered_df, all_checks_df)
