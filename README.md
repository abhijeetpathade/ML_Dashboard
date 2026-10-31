# Integrated Machine Learning Dashboard

An interactive web application built with **Python and Flask** that brings 11 machine learning experiments into one dashboard. Users can select an experiment, enter input values, run a model, and view predictions, evaluation metrics, and visualizations.

## Project Overview

| Item | Details |
|---|---|
| Project title | Integrated Machine Learning Dashboard |
| Project type | College mini project |
| Application | Web-based machine learning dashboard |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Machine learning | Scikit-learn |
| Data handling | Pandas, NumPy |
| Visualizations | Matplotlib |
| Number of experiments | 11 |
| Data source | CSV files in the `datasets/` folder |

## Experiments

| No. | Experiment name | Algorithm / task | Dataset |
|---|---|---|---|
| 01 | Understanding Any Dataset Like a Data Scientist | Dataset exploration and descriptive analysis | `student_dataset.csv` |
| 02 | Simple Linear Regression | Simple Linear Regression | `Placement.csv` |
| 03 | Multiple Linear Regression | Multiple Linear Regression | `Advertising.csv` |
| 04 | Binary Logistic Regression | Binary Logistic Regression | `Placement_Y_N.csv` |
| 05 | Multiclass Logistic Regression | Multiclass Logistic Regression | Iris dataset |
| 06 | Decision Tree Classification | Decision Tree | `heart_data.csv` |
| 07 | K-Nearest Neighbors (KNN) Classification | KNN | `diabetes_dataset.csv` |
| 08 | Naive Bayes Classification | Naive Bayes text classification | `college_messages.csv` |
| 09 | Linear Support Vector Machine (SVM) | Linear SVM | `data_banknote_authentication.csv` |
| 10 | Non-Linear Support Vector Machine (RBF Kernel) | RBF SVM | `data_banknote_authentication.csv` |
| 11 | Random Forest Classification | Random Forest | `diabetes.csv` |

## Features

- One dashboard containing all 11 experiments.
- Separate page and input form for each experiment.
- Dataset name displayed on each experiment page.
- Input and expected-output guidance.
- Prediction results and model evaluation metrics.
- Regression plots, classification confusion matrices, and feature-importance charts where applicable.
- Dataset-based numeric input validation.
- Algorithm explanations for each experiment.
- Dashboard overview cards.
- About Project page.

## Input Fields and Expected Outputs

| Experiment | Input values | Expected output |
|---|---|---|
| Simple Linear Regression | CGPA | Predicted placement package |
| Multiple Linear Regression | TV and Newspaper advertising budgets | Predicted sales |
| Binary Logistic Regression | CGPA and IQ | Placed / Not placed |
| Multiclass Logistic Regression | Sepal length, sepal width, petal length, petal width | Iris species |
| Decision Tree Classification | Age, resting blood pressure, cholesterol, maximum heart rate, oldpeak | Heart-disease class (0 or 1) |
| KNN Classification | Pregnancies, Glucose, Blood Pressure, BMI, Age | Diabetes class (0 or 1) |
| Naive Bayes Classification | Message text | Academic, Promotional, or Placement |
| Linear SVM | Variance, skewness, curtosis, entropy | Genuine / Counterfeit |
| Non-Linear SVM | Variance, skewness, curtosis, entropy | Genuine / Counterfeit |
| Random Forest Classification | Pregnancies, Glucose, Blood Pressure, BMI, Age | Diabetes class (0 or 1) |

Experiment 01 performs dataset exploration and does not require prediction inputs. For Decision Tree, KNN, and Random Forest, the dashboard displays five input fields; remaining features are filled using dataset medians.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Backend web framework |
| HTML, CSS, JavaScript | Frontend |
| Pandas and NumPy | Data handling |
| Scikit-learn | Machine learning algorithms and evaluation |
| Matplotlib | Visualizations |

## Project Structure

```text
Integrated-ML-Dashboard/
├── app.py
├── requirements.txt
├── RUN_DASHBOARD.bat
├── .gitignore
├── README.md
├── datasets/
├── models/
├── templates/
└── notebooks/
```

## How to Run

1. Install Python 3.11 or newer.
2. Download or clone this repository.
3. Open the project folder in VS Code.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   python app.py
   ```

   Alternatively, double-click `RUN_DASHBOARD.bat`.

6. Open `http://127.0.0.1:5000` in your browser.

Keep the terminal open while using the dashboard. Press `Ctrl + C` to stop the Flask server.

## Models Folder

The `models/` folder contains selected model and vectorizer files collected from the original experiment repository. They are included as reference artifacts. The current dashboard trains models from the CSV datasets when experiments are run; it does not automatically load these saved files. Verify dataset, feature order, preprocessing, and library versions before using a saved artifact.

**Security note:** Only load `.pkl` or `.joblib` files from trusted sources.

## Evaluation Notes

- Metrics are calculated using a held-out test split.
- Predictions depend on the values entered by the user.
- For Decision Tree, KNN, and Random Forest, hidden features use dataset medians, so predictions are conditional on those default values.
- Experiment 01 is dataset analysis, not a prediction experiment.

## Future Enhancements

- Save prediction history between sessions.
- Export prediction reports.
- Compare compatible algorithms using common datasets and evaluation metrics.
- Add validated loading of saved model artifacts.

## Author

Developed as a college mini project.
