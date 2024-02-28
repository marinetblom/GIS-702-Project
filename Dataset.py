import pandas as pd
import numpy as np

# Try different encodings until you find the one that works
encodings_to_try = ["utf-8", "latin1", "cp1252"]

for encoding in encodings_to_try:
    try:
        data = pd.read_csv(
            r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\0021178a3 5min.ttx",
            delimiter=",",
            encoding=encoding,
        )
        print("File loaded successfully with encoding:", encoding)
        break
    except UnicodeDecodeError:
        print("Failed to read with encoding:", encoding)

# Load the file with the correct delimiter and parse the "Date" column as datetime
data = pd.read_csv(
    r"C:\Users\Dell 5401\Documents\Honours\GIS 702 Research Project\GIS-702-Project\0021178a3 5min.ttx",
    delimiter="\t",
    encoding="latin1",
)

# Replace commas with periods and remove thousand separators for numeric columns
# numeric_columns = ['Speed', 'WindDir', 'Gust', 'Temperature', 'Humidity', 'Pressure', 'Rain']
# for col in numeric_columns:
#     data[col] = data[col].astype(str).str.replace(',', '.').str.replace(' ', '')
#     # Identify rows with non-numeric values
#     non_numeric_mask = ~data[col].str.match(r'^-?\d*\.?\d+$')  # Regex to match numeric values
#     if non_numeric_mask.any():
#         print(f"Non-numeric values found in column '{col}':")
#         print(data.loc[non_numeric_mask, col])
#         # Replace non-numeric values with NaN
#         data.loc[non_numeric_mask, col] = np.nan

# Convert the cleaned strings to float data type
# data[numeric_columns] = data[numeric_columns].astype(float)

# Now you can work with the data as needed
# For example, print the first few rows to inspect the data
print(data.head())
