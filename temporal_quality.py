# temporal_quality.py
import pandas as pd
import matplotlib.pyplot as plt


def check_out_of_order_timestamps(df):
    if not df["DateT"].is_monotonic_increasing:
        print("Data contains out-of-order timestamps.")
    else:
        print("Data contains NO out-of-order timestamps")


def check_duplicate_timestamps(df):
    duplicate_timestamps = df["DateT"].duplicated().sum()
    if duplicate_timestamps > 0:
        print(f"Data contains {duplicate_timestamps} duplicate timestamps.")
    else:
        print("No duplicate timestamps")


def check_temporal_quality(df):
    # Set DateT as the index
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
