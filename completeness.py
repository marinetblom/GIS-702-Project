# completeness.py
import missingno as msno
import matplotlib.pyplot as plt


def check_missing_data(df):
    missing_values = df.isnull().sum()
    print("Missing values per column:\n", missing_values)

    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print("Percentage of missing values per column:\n", missing_percentage)

    # Visualize missing data using missingno
    msno.matrix(df)
    msno.dendrogram(df)
    plt.show()
