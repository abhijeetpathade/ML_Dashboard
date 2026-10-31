# Integrated Machine Learning Dashboard

An interactive web application built with **Python and Flask** that
brings 11 machine learning practical experiments into one dashboard.
Users can select an experiment, enter the required input values, run the
model, and view predictions, evaluation metrics, and visualizations.

## Project Overview

  Item                    Details
  ----------------------- -------------------------------------------------
  Project title           Integrated Machine Learning Dashboard
  Project type            College mini project
  Application             Web-based machine learning dashboard
  Backend                 Python, Flask
  Frontend                HTML, CSS, JavaScript
  Machine learning        Scikit-learn
  Data handling           Pandas, NumPy
  Visualizations          Matplotlib
  Number of experiments   11
  Data source             CSV datasets included in the `datasets/` folder

## Experiments

  --------------------------------------------------------------------------------------------
                    No. Experiment name  Algorithm / task Dataset
  --------------------- ---------------- ---------------- ------------------------------------
                     01 Understanding    Dataset          `student_dataset.csv`
                        Any Dataset Like exploration and  
                        a Data Scientist descriptive      
                                         analysis         

                     02 Simple Linear    Simple Linear    `Placement.csv`
                        Regression       Regression       

                     03 Multiple Linear  Multiple Linear  `Advertising.csv`
                        Regression       Regression       

                     04 Binary Logistic  Binary Logistic  `Placement_Y_N.csv`
                        Regression       Regression       

                     05 Multiclass       Multiclass       Iris dataset
                        Logistic         Logistic         
                        Regression       Regression       

                     06 Decision Tree    Decision Tree    `heart_data.csv`
                        Classification                    

                     07 K-Nearest        KNN              `diabetes_dataset.csv`
                        Neighbors (KNN)                   
                        Classification                    

                     08 Naive Bayes      Naive Bayes text `college_messages.csv`
                        Classification   classification   

                     09 Linear Support   Linear SVM       `data_banknote_authentication.csv`
                        Vector Machine                    
                        (SVM)                             

                     10 Non-Linear       RBF SVM          `data_banknote_authentication.csv`
                        Support Vector                    
                        Machine (RBF                      
                        Kernel)                           

                     11 Random Forest    Random Forest    `diabetes.csv`
                        Classification                    
  --------------------------------------------------------------------------------------------

## Features

-   One dashboard containing all 11 experiments.
-   Separate page and input form for each experiment.
-   Dataset name displayed on each experiment page.
-   Input and expected-output guidance to explain what values are
    entered and what the model predicts.
-   Prediction results and model evaluation metrics.
-   Visualizations, including regression actual-versus-predicted plots,
    classification confusion matrices, and tree-model feature-importance
    charts where applicable.
-   Dataset-based numeric input validation.
-   Algorithm explanations for each experiment.
-   Dashboard overview cards showing experiment totals by category.
-   About Project page with project and technology information.

## Input Fields and Predictions

  -----------------------------------------------------------------------
  Experiment              User input              Expected output
  ----------------------- ----------------------- -----------------------
  Simple Linear           CGPA                    Predicted placement
  Regression                                      package

  Multiple Linear         TV and Newspaper        Predicted sales
  Regression              advertising budgets     

  Binary Logistic         CGPA and IQ             Placed / Not placed
  Regression                                      

  Multiclass Logistic     Sepal length, sepal     Iris species: Setosa,
  Regression              width, petal length,    Versicolor, or
                          petal width             Virginica

  Decision Tree           Age, resting blood      Heart-disease class (0
  Classification          pressure (`trestbps`),  or 1)
                          cholesterol (`chol`),   
                          maximum heart rate      
                          (`thalach`), and        
                          `oldpeak`               

  KNN Classification      Pregnancies, Glucose,   Diabetes class (0 or 1)
                          Blood Pressure, BMI,    
                          and Age                 

  Naive Bayes             Message text            Academic, Promotional,
  Classification                                  or Placement

  Linear SVM              Variance, skewness,     Banknote class (Genuine
                          curtosis, and entropy   / Counterfeit)

  Non-Linear SVM          Variance, skewness,     Banknote class (Genuine
                          curtosis, and entropy   / Counterfeit)

  Random Forest           Pregnancies, Glucose,   Diabetes class (0 or 1)
  Classification          Blood Pressure, BMI,    
                          and Age                 
  -----------------------------------------------------------------------

**Experiment 01** is for dataset understanding and does not require
prediction inputs. For Decision Tree, KNN, and Random Forest, the
dashboard exposes five practical input fields; remaining model features
are filled using medians calculated from the dataset.

## Project Structure

``` text
Integrated-ML-Dashboard/
├── app.py
├── requirements.txt
├── RUN_DASHBOARD.bat
├── .gitignore
├── README.md
├── datasets/
│   ├── student_dataset.csv
│   └── other experiment CSV files
├── models/
│   ├── binary_logistic_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.joblib
│   ├── linear_svm_model.pkl
│   ├── multiclass_logistic_model.pkl
│   ├── multiple_linear_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── naive_bayes_vectorizer.pkl
│   ├── nonlinear_svm_model.pkl
│   ├── random_forest_model.joblib
│   ├── simple_linear_regression_model.pkl
│   └── MODEL_CATALOG.md
├── templates/
│   ├── home.html
│   ├── experiment.html
│   └── about.html
└── notebooks/
    └── experiment notebooks
```

## Requirements

-   Python 3.11 or newer
-   pip
-   A modern web browser

## Run on Windows

1.  Download and extract the project.

2.  Open the extracted project folder in VS Code.

3.  Open the VS Code terminal in the project folder.

4.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

5.  Start the application:

    ``` bash
    python app.py
    ```

    Alternatively, double-click `RUN_DASHBOARD.bat`.

6.  Open the dashboard in your browser:

    `http://127.0.0.1:5000`

Keep the terminal open while using the dashboard. Stop the Flask server
with `Ctrl + C`.

## Models Folder Note

The `models/` folder contains selected serialized model and vectorizer
files collected from the supplied machine learning experiment
repository. They are included as project artifacts/reference files.
**The current dashboard trains models from the CSV datasets when
experiments are run; it does not automatically load these saved model
files.** Before using a saved `.pkl` or `.joblib` artifact, verify that
its dataset, feature order, preprocessing steps, and library versions
match the application.

Only load pickle/joblib files from sources you trust because
deserializing untrusted files can execute code.

## Evaluation and Interpretation

-   Model metrics are calculated using a held-out test split.
-   Predictions depend on the input values entered by the user.
-   For Decision Tree, KNN, and Random Forest, unexposed input features
    are filled with dataset medians. Their predictions are therefore
    conditional on those default values.
-   Experiment 01 presents dataset inspection and descriptive analysis
    rather than a prediction.

## Future Enhancements

-   Save prediction history between application sessions.
-   Export individual prediction reports.
-   Compare compatible algorithms using common evaluation datasets and
    metrics.
-   Add model-loading support after validating the saved artifacts
    against the dashboard pipeline.

## Author

**Developed as a college mini project.**
