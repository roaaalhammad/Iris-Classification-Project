# Iris Classification Project

This project implements a complete machine learning pipeline for
classifying Iris flower species using the Iris dataset.\
It includes data loading, cleaning, visualization, model training,
prediction, and both console and GUI-based user interfaces.

##  Project Structure

    iris-project/
    │
    ├── src/
    │   ├── data_loader.py      # Loads and cleans the Iris dataset
    │   ├── exploration.py      # Generates EDA plots (histogram, scatter, boxplot)
    │   ├── model_trainer.py    # Trains and evaluates the KNN model
    │   ├── predictor.py        # Predicts species for new input samples
    │   ├── iris_model.pkl      # Saved machine learning model (generated after training)
    │   └── __init__.py         # Marks src as a Python package
    │
    ├── iris.png                # Iris App image used in the Jupyter Notebook
    ├── main.py                 # Complete console application (pipeline + terminal interface)
    ├── app_tk.py               # Tkinter GUI for interactive species prediction
    └── notebook.ipynb          # Full project report, explanation, and visualizations

##  Project Overview

### 1. Data Loading & Cleaning

`data_loader.py` loads the Iris dataset, converts it to a DataFrame,
removes duplicates, and provides basic dataset summaries.

### 2. Exploratory Data Analysis (EDA)

`exploration.py` generates histograms, scatter plots, and box plots to
help understand feature distributions and relationships.

### 3. Model Training

`model_trainer.py` trains a K-Nearest Neighbors (KNN) model and
evaluates its performance using standard ML metrics.

### 4. Prediction Module

`predictor.py` predicts the Iris species for new measurements.

### 5. Console Application

`main.py` runs the full pipeline and provides a terminal-based
prediction interface.

### 6. GUI Application

`app_tk.py` provides an interactive Tkinter GUI for users to input
measurements and receive predictions visually.

### 7. Notebook Report

`notebook.ipynb` documents the entire workflow, including explanations,
plots, and demonstration steps.

##  How to Run

### Run Console Application

    python main.py

### Run GUI Application

    python app_tk.py

##  Requirements

-   pandas
-   scikit-learn
-   matplotlib
-   joblib
-   tkinter


## Authors

-   Roaa Mohammad Alhammad
-   Rema Abdulrahman Alluhaymid
-   Manar Abbas Almutairi
