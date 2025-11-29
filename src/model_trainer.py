from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.data_loader import IrisData

class ModelTrainer:
    def train_and_evaluate(self):
        # Load the cleaned Iris dataset from data_loader
        iris = IrisData().get_clean_iris()

        # Select features and target variable
        features = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
        X = iris[features].values
        y = iris["species_id"].values

        # Get class names safely (in case 'species' is not categorical)
        try:
            names = list(iris["species"].cat.categories)
        except Exception:
            names = ["setosa", "versicolor", "virginica"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize and train the KNN model
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Evaluate the model's performance
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=names)

        # Display evaluation results
        print("=== Model Evaluation Results ===")
        print(f"Accuracy: {acc * 100:.2f}%")
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)

        # Example: make a prediction for a single sample
        sample = [[5.1, 3.5, 1.4, 0.2]]
        pred = model.predict(sample)
        print("\nPredicted Class:", names[pred[0]])

        # Return the trained model for use in the main application
        return model


# Run only when executed directly, not when imported by another file
if __name__ == "__main__":
    trainer = ModelTrainer()
    model = trainer.train_and_evaluate()

    import joblib
    joblib.dump(model, "iris_model.pkl")
    print("iris_model.pkl saved successfully")



