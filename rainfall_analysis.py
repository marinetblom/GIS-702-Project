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


# Function to apply POT filter based on a direct threshold value
def apply_pot_filter(df, threshold_value=3):
    df = df[df["Rain"] > threshold_value]
    return df


# Function to identify max rainfall events using moving average (optimized with overlapping windows)
def identify_moving_average_max_rainfall_optimized(df, window_size_minutes):
    max_events = []
    window_size = window_size_minutes // 5  # Since the data is measured every 5 minutes

    # Group the data by year
    for year, year_df in df.groupby(df["DateT"].dt.year):
        print(f"\nProcessing year: {year} with {len(year_df)} data points.")

        # Set DateT as index for rolling calculation
        year_df = year_df.set_index("DateT")

        # Resample the data to ensure consistent 5-minute intervals, filling missing timestamps with NaN
        year_df = year_df.resample("5min").asfreq()  # Fix for the deprecation warning

        print(f"After resampling, the year has {len(year_df)} data points.")

        # Optionally, fill NaNs in the Rain column (no inplace=True to avoid the warning)
        year_df["Rain"] = year_df["Rain"].fillna(0)  # Fix for the inplace warning

        # Calculate rolling sum of Rain for the given window size (overlapping windows)
        year_df["Rain_MA"] = (
            year_df["Rain"].rolling(window=window_size, min_periods=1).sum()
        )

        # Check if the rolling calculation worked correctly
        print(f"Rolling window calculation complete for year: {year}")

        # Find the time window with the highest rolling sum
        max_rainfall_idx = year_df["Rain_MA"].idxmax()
        max_rainfall_row = year_df.loc[max_rainfall_idx]
        max_rain = max_rainfall_row["Rain_MA"]

        # Define start and end of the maximum rolling window
        end_time = max_rainfall_idx
        start_time = end_time - pd.Timedelta(minutes=window_size_minutes)

        # Print the details of the maximum window
        print(
            f"Year: {year}, Max Window: {start_time} to {end_time}, Total Rain: {max_rain} mm"
        )

        # Filter original data within the identified window
        window_df = year_df[
            (year_df.index >= start_time) & (year_df.index <= end_time)
        ].reset_index()

        # Find the row with the maximum rainfall in this window
        max_event = window_df.loc[window_df["Rain"].idxmax()]
        max_events.append(max_event)

    # Return a DataFrame of the identified maximum events
    max_events_df = pd.DataFrame(max_events)
    return max_events_df
