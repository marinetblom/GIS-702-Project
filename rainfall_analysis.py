import pandas as pd
import matplotlib.pyplot as plt


# Function to identify annual maximum rainfall
def identify_annual_max_rainfall(df):
    return df.loc[df.groupby(df["DateT"].dt.year)["Rain"].idxmax()]


# Function to plot annual maximum rainfall
def plot_annual_max_rainfall(df):
    df["Year"] = df["DateT"].dt.year
    plt.figure(figsize=(12, 6))
    plt.bar(df["Year"], df["Rain"], color="skyblue")
    plt.title("Annual Maximum Rainfall (5-min interval)")
    plt.xlabel("Year")
    plt.ylabel("Maximum Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Function to apply POT filter
def apply_pot_filter(df, threshold_percent=90):
    df = df[df["Rain"] > 0]
    threshold_value = df["Rain"].quantile(threshold_percent / 100)
    pot_df = df[df["Rain"] >= threshold_value]
    return pot_df


# Function to identify max rainfall events using moving average (optimized)
def identify_moving_average_max_rainfall_optimized(df, window_size_minutes):
    max_events = []
    window_freq = f"{window_size_minutes}min"

    # Group the data by year
    for year, year_df in df.groupby(df["DateT"].dt.year):
        # Resample the data to get rainfall sums for each window
        resampled_df = (
            year_df.set_index("DateT")
            .resample(window_freq)
            .sum(min_count=1)
            .reset_index()
        )

        # Find the window with the highest total rainfall
        max_row = resampled_df.loc[resampled_df["Rain"].idxmax()]
        start_time = max_row["DateT"]
        end_time = start_time + pd.Timedelta(minutes=window_size_minutes)

        # Print the details of the maximum window
        print(
            f"Year: {year}, Max Window: {start_time} to {end_time}, Total Rain: {max_row['Rain']} mm"
        )

        # Filter original data within the identified window
        window_df = year_df[
            (year_df["DateT"] >= start_time) & (year_df["DateT"] <= end_time)
        ]

        # Find the row with the maximum rainfall in this window
        max_event = window_df.loc[window_df["Rain"].idxmax()]
        max_events.append(max_event)

    max_events_df = pd.DataFrame(max_events)
    return max_events_df
