# Sleep Disorder Classification using Random Forest in PySpark

This repository contains a machine learning model for classifying sleep disorders based on various health features. The model is built using PySpark and Random Forest Classifier, with various evaluation metrics like accuracy, F1 score, precision, and recall. Cross-validation and hyperparameter tuning are also applied for optimal model performance.

## Project Overview

The dataset consists of features related to individuals' health, such as blood pressure, age, physical activity level, and heart rate, along with a target variable indicating whether the individual has a sleep disorder. The model uses these features to predict the likelihood of sleep disorders.

### Features Used:
- Ordinal BMI
- Diastolic Blood Pressure
- Systolic Blood Pressure
- Sleep Duration
- Physical Activity Level
- Age
- Occupation Vector
- Daily Steps
- Heart Rate
- Stress Level

## Main Components:
1. **Data Preprocessing**: Preparing and assembling the features for training.
2. **Random Forest Classifier**: Training a Random Forest model to predict sleep disorders.
3. **Hyperparameter Tuning**: Using `TrainValidationSplit` to find optimal hyperparameters.
4. **Cross-Validation**: Evaluating model performance with different metrics.
5. **Evaluation Metrics**: Calculating accuracy, F1 score, precision, recall, and confusion matrix.
6. **Visualization**: Plotting confusion matrices to visualize classification performance.

## Requirements

- PySpark 3.x
- pandas
- numpy
- matplotlib
- seaborn

## Setup Instructions

1. **Install the required libraries:**

    ```bash
    pip install pyspark pandas numpy matplotlib seaborn
    ```

2. **Download or Clone this repository:**

    ```bash
    git clone https://github.com/yourusername/sleep-disorder-classification.git
    ```

3. **Run the Jupyter notebook**:

    Open the Jupyter notebook (`sleep_disorder_classification.ipynb`) and run the code cells sequentially.

    ```bash
    jupyter notebook
    ```

## Model Explanation

### Random Forest Classifier:
The Random Forest algorithm is a powerful ensemble method that uses multiple decision trees to make predictions. Each tree is trained on a random subset of the data, and predictions are made by aggregating the predictions of all trees in the forest.

### Hyperparameter Tuning:
The hyperparameters of the Random Forest classifier are optimized using `TrainValidationSplit` to select the best model based on evaluation metrics.

### Evaluation Metrics:
- **Accuracy**: Proportion of correct predictions.
- **F1 Score**: Harmonic mean of precision and recall, useful for imbalanced datasets.
- **Precision and Recall**: Measures of the model's ability to predict positive cases and identify all positive instances, respectively.
- **Confusion Matrix**: Visualizes the performance of the model, highlighting the true positives, false positives, true negatives, and false negatives.

## Conclusion

This repository provides an end-to-end solution for building and evaluating a machine learning model to predict sleep disorders. By using PySpark, it leverages distributed computing to handle large datasets efficiently. The model's performance is evaluated using multiple metrics, and hyperparameter tuning is done to ensure the best possible model performance.

## Future Improvements

- Experiment with other classification algorithms like XGBoost or SVM.
- Incorporate more features or data sources to improve prediction accuracy.
- Deploy the model for real-time predictions using a web framework (Flask/Django).



