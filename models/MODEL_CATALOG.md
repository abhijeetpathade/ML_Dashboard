# Model Catalog

This folder contains one selected saved model for each model-based experiment in the dashboard, plus the Naive Bayes text vectorizer required for message preprocessing.

| Dashboard experiment | File |
|---|---|
| Simple Linear Regression | `simple_linear_regression_model.pkl` |
| Multiple Linear Regression | `multiple_linear_regression_model.pkl` |
| Binary Logistic Regression | `binary_logistic_model.pkl` |
| Multiclass Logistic Regression | `multiclass_logistic_model.pkl` |
| Decision Tree Classification | `decision_tree_model.pkl` |
| KNN Classification (diabetes dataset) | `knn_model.joblib` |
| Naive Bayes Classification | `naive_bayes_model.pkl` and `naive_bayes_vectorizer.pkl` |
| Linear SVM | `linear_svm_model.pkl` |
| Non-Linear SVM (RBF) | `nonlinear_svm_model.pkl` |
| Random Forest Classification (diabetes dataset) | `random_forest_model.joblib` |

Experiment 01 (Dataset Understanding) does not require a trained model.

The extra `*_v2`, breast-cancer KNN, real-estate regression, and Random Forest glucose-regression artifacts were left out because they are alternate versions or belong to different datasets/assignments rather than the dashboard's selected experiment setup.

**Compatibility note:** These are archived artifacts from the supplied experiment ZIP. The current dashboard trains models from its included CSV files and does not automatically load these saved files. Before loading any artifact, verify its feature order, preprocessing, dataset, and scikit-learn version.
