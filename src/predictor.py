class Predictor:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.species_names = ['setosa', 'versicolor', 'virginica']
        self.feature_names = feature_names or [
            'sepal length (cm)', 
            'sepal width (cm)', 
            'petal length (cm)', 
            'petal width (cm)'
        ]
    
    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        """Predict species based on measurements."""
        try:
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
            prediction = self.model.predict(input_data)
            species_id = prediction[0]
            return self.species_names[species_id]
        except Exception as e:
            return f"Error in prediction: {str(e)}"
    
    def get_feature_names(self):
        """Return the feature names for reference."""
        return self.feature_names