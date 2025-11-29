from src.data_loader import IrisData
from src.model_trainer import ModelTrainer
from src.predictor import Predictor

class IrisApp:
    def __init__(self):
        self.predictor = None

    def run(self):
        print("Starting Iris Flower Classification Project...\n")

        # 1. Loading and cleaning data
        print("1. Loading and cleaning data...")
        data_loader = IrisData()
        clean_data = data_loader.get_clean_iris()

        # Print dataset information
        data_loader.print_info(clean_data)

        print("\n2. Training machine learning model...")
        trainer = ModelTrainer()
        model = trainer.train_and_evaluate()

        print("\n3. Creating predictor...")
        self.predictor = Predictor(model)

        print("\n4. Testing prediction system...")
        self.test_predictions()

        print("\nProject integration completed successfully!")

        self.start_terminal_interface()

    def test_predictions(self):
        test_samples = [
            [5.1, 3.5, 1.4, 0.2],
            [6.0, 2.7, 5.1, 1.6],
            [5.7, 2.8, 4.1, 1.3]
        ]

        for i, sample in enumerate(test_samples, 1):
            result = self.predictor.predict_species(*sample)
            print(f"Test {i}: {sample} -> {result}")

    def start_terminal_interface(self):
        print("\nTerminal Testing Interface (Type 'quit' to exit):")
        print("Enter measurements as: sepal_length sepal_width petal_length petal_width")

        while True:
            try:
                user_input = input("\nEnter measurements: ").strip()
                if user_input.lower() == 'quit':
                    print("Thank you for using Iris Classifier!")
                    break

                if user_input:
                    measurements = [float(x) for x in user_input.split()]
                    if len(measurements) == 4:
                        result = self.predictor.predict_species(*measurements)
                        print(f"Predicted species: {result}")
                    else:
                        print("Please enter exactly 4 numbers separated by spaces")
                else:
                    print("Please enter some values")

            except ValueError:
                print("Please enter valid numbers (example: 5.1 3.5 1.4 0.2)")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    app = IrisApp()
    app.run()