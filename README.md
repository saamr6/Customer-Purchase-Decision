# House Purchase Decision Classification

Predicting house purchase decisions using Logistic Regression (Scikit-learn) and a Neural Network (TensorFlow/Keras) on a global dataset. Includes EDA, preprocessing, SMOTE for imbalance handling, and model evaluation.

## Description
This repository contains a Jupyter Notebook (`house_purchase_classification.ipynb`) that analyzes a global house purchase dataset (`global_house_purchase_dataset.csv`). The primary goal is to predict the customer's purchase decision (binary classification: 0 or 1) based on various property, financial, and customer attributes.

The project demonstrates data preprocessing techniques, exploratory data analysis (EDA), handling class imbalance using SMOTE, and compares the performance of two different classification models:
1.  **Logistic Regression** (using Scikit-learn)
2.  **Artificial Neural Network (ANN)** / Multi-layer Perceptron (MLP) (using TensorFlow/Keras)

## Dataset
The dataset (`global_house_purchase_dataset.csv`) contains information about properties and potential buyers, including features like:
* Geographical: `country`, `city`
* Property Details: `property_type`, `furnishing_status`, `property_size_sqft`, `constructed_year`, `previous_owners`, `rooms`, `bathrooms`, `stories`
* Customer Financials: `customer_salary`, `loan_amount`, `loan_tenure_years`, `monthly_expenses`, `down_payment`, `emi_to_income_ratio`
* Ratings/Scores: `satisfaction_score`, `neighbourhood_rating`, `connectivity_score`
* Target Variable: `decision` (0 or 1)

*(Note: `property_id` was dropped during preprocessing).*

## Analysis and Modeling Steps
1.  **Import Libraries:** Pandas, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn (SMOTE), TensorFlow/Keras.
2.  **Load Data & Initial Cleaning:** Load dataset and drop the `property_id` column.
3.  **Exploratory Data Analysis (EDA):** Visualize correlations between numerical features and all features (after encoding) using heatmaps.
4.  **Data Preprocessing:**
    * Convert object-type columns (categorical features) into numerical codes.
    * Split data into features (X) and target (y).
    * Split data into training (80%) and testing (20%) sets.
    * Apply `StandardScaler` to scale numerical features.
    * Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to the *training data* to handle class imbalance in the `decision` variable.
5.  **Model 1: Logistic Regression:**
    * Train a `LogisticRegression` model (Scikit-learn) on the scaled and oversampled training data.
    * Make predictions on the scaled test data.
    * Evaluate using a `classification_report`.
6.  **Model 2: Neural Network (TensorFlow/Keras):**
    * Define a Sequential model architecture with Dense layers, ReLU activation, Dropout for regularization, and a final Sigmoid activation for binary classification.
    * Compile the model using Adam optimizer and binary crossentropy loss.
    * Implement `EarlyStopping` to prevent overfitting.
    * Train the model on the scaled and oversampled training data, validating on the scaled test data.
    * Visualize training/validation loss and accuracy curves.
    * Evaluate the final model on the test set.
    * Make predictions and generate a `classification_report`.

## Tools Used
* Python 3
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn (`train_test_split`, `LogisticRegression`, `StandardScaler`, `metrics`)
* Imbalanced-learn (`SMOTE`)
* TensorFlow / Keras (`Sequential`, `Dense`, `Dropout`, `Input`, `Adam`, `EarlyStopping`)
* Jupyter Notebook

## How to Run
1.  Clone this repository:
    ```bash
    git clone https://github.com/saamr6/Customer-Purchase-Decision
    ```
2.  Navigate to the cloned directory.
3.  Ensure you have Python installed, along with the necessary libraries:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow notebook
    ```
4.  Make sure the dataset `global_house_purchase_dataset.csv` is present in the same directory.
5.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
6.  Open `house_purchase_classification.ipynb` and run the cells.

## Results
The notebook provides a comparison between the Logistic Regression and Neural Network models based on their classification reports (precision, recall, f1-score, accuracy) on the test set. The Neural Network model, trained on SMOTE-balanced data, demonstrated strong performance in classifying purchase decisions. Detailed metrics and training plots are available within the notebook.
