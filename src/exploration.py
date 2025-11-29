import matplotlib.pyplot as plt
from src.data_loader import IrisData


class DataExplorer:

    def __init__(self):
        # Load and clean data once when the object is created
        iris = IrisData()
        self.df = iris.get_clean_iris()

    # Histogram plot
    def plot_histograms(self):
        self.df.hist(figsize=(8, 6), bins=10)
        plt.suptitle("Histograms of Iris Dataset ")
        plt.tight_layout()
        plt.show()

    # Scatter plot
    def plot_scatter(self):
        species_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
        colors = self.df["species"].map(species_map)

        plt.scatter(
            self.df["petal length (cm)"],
            self.df["petal width (cm)"],
            c=colors,
            cmap="viridis",
        )
        plt.title("Petal Length vs Petal Width (by Species)")
        plt.xlabel("Petal Length (cm)")
        plt.ylabel("Petal Width (cm)")

        cbar = plt.colorbar()
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["setosa", "versicolor", "virginica"])
        cbar.set_label("Species")

        plt.show()

    # Box plot
    def plot_box(self):
        self.df.boxplot()
        plt.title("Box Plot")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


# Run the plots only when this file is opened directly, not when imported by another file.
if __name__ == "__main__":
    explorer = DataExplorer()
    explorer.plot_histograms()
    explorer.plot_scatter()
    explorer.plot_box()
