# Import libraries
import pandas as pd
from sklearn.datasets import load_iris


class IrisData:
    """
    Class to load, clean, and return the Iris dataset.
    This component is responsible for accessing the dataset,
    converting it into a DataFrame, adding target labels,
    and performing basic cleaning operations.
    """

    def get_clean_iris(self) -> pd.DataFrame:
        """
        Load the Iris dataset, build a DataFrame, clean duplicates,
        and return the processed dataset.
        """

        # Step 1: Import the Iris dataset from sklearn
        iris = load_iris()

        # Understanding the variables:
        # sepal length (cm): length of the flower's sepal
        # sepal width (cm): width of the flower's sepal
        # petal length (cm): length of the flower's petal
        # petal width (cm): width of the flower's petal
        # species: type of Iris flower (Setosa, Versicolor, Virginica)

        # Step 2: Convert raw data into a pandas DataFrame
        df = pd.DataFrame(iris.data, columns=iris.feature_names)

        # Add target columns
        df["species_id"] = iris.target
        df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

        # Remove duplicate rows (if any)
        df = df.drop_duplicates()

        return df

    def print_info(self, df: pd.DataFrame):
        """Print dataset information."""
        print("Shape:", df.shape)
        print(df.head())

        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nDuplicate rows:")
        print(df.duplicated().sum())


# Run this file directly to preview dataset information
if __name__ == "__main__":
    iris = IrisData()
    df = iris.get_clean_iris()
    iris.print_info(df)
