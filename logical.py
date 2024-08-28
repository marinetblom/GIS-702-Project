# logical.py
import pandas as pd


# Summary statistics function with export to CSV
def summarize_statistics(filtered_df, file_path):
    pd.set_option("display.float_format", lambda x: "%.3f" % x)
    summary = filtered_df.describe()
    print(summary)
    summary.to_csv(file_path)
