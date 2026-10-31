import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import io, base64
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree

app = Flask(__name__)
DATA = os.path.join(os.path.dirname(__file__), "datasets")

# Keeps the five most recent successful actions while the Flask server is running.
RECENT_ACTIVITY = []

EXPERIMENTS = [
 {"id":"dataset","no":"01","name":"Understanding a Dataset","category":"Data Analysis","desc":"Inspect a real lab dataset, its columns, records and summary statistics.","dataset":"student_dataset.csv — Student Academic Dataset","input_info":'No manual input is required. The application loads the sample student dataset automatically.',"output_info":'Dataset dimensions, sample records and descriptive statistics.',"kind":"analysis"},
 {"id":"slr","no":"02","name":"Simple Linear Regression","category":"Regression","desc":"Estimate placement package from CGPA.","dataset":"Placement.csv — Student CGPA and Package Dataset","input_info":'Enter the student CGPA, using a value within the range of the dataset.',"output_info":'Predicted placement package.',"kind":"regression"},
 {"id":"mlr","no":"03","name":"Multiple Linear Regression","category":"Regression","desc":"Estimate sales using TV and newspaper advertising budgets.","dataset":"Advertising.csv — Advertising and Sales Dataset","input_info":'Enter TV and Newspaper advertising budgets using the same units as the CSV.',"output_info":'Predicted Sales value.',"kind":"regression"},
 {"id":"logbin","no":"04","name":"Logistic Regression — Binary","category":"Classification","desc":"Predict placement status using CGPA and IQ.","dataset":"Placement_Y_N.csv — Student Placement Dataset","input_info":'Enter the student CGPA and IQ score.',"output_info":'Predicted placement status: Placed or Not placed.',"kind":"classification"},
 {"id":"logmulti","no":"05","name":"Logistic Regression — Multiclass","category":"Classification","desc":"Predict Iris flower species using four measurements.","dataset":"Iris Dataset — built into scikit-learn","input_info":'Enter sepal length, sepal width, petal length and petal width in centimetres.',"output_info":'Predicted Iris species: Setosa, Versicolor or Virginica.',"kind":"classification"},
 {"id":"tree","no":"06","name":"Decision Tree Classifier","category":"Classification","desc":"Predict heart-disease target from selected clinical measurements.","dataset":"heart_data.csv — Heart Disease Dataset","input_info":'Enter Age, resting blood pressure (trestbps), cholesterol (chol), maximum heart rate (thalach) and ST depression (oldpeak). Other features use dataset medians.',"output_info":'Predicted heart-disease class: disease present (1) or absent (0).',"kind":"classification"},
 {"id":"knn","no":"07","name":"KNN Classifier","category":"Classification","desc":"Predict diabetes outcome using the lab diabetes dataset.","dataset":"diabetes_dataset.csv — Pima Indians Diabetes Dataset","input_info":'Enter Pregnancies, Glucose, BloodPressure, BMI and Age. Other features use dataset medians.',"output_info":'Predicted diabetes outcome: diabetes detected (1) or not detected (0).',"kind":"classification"},
 {"id":"nb","no":"08","name":"Naive Bayes Classifier","category":"Text Classification","desc":"Classify a college message as Academic, Promotional or Placement.","dataset":"college_messages.csv — College Messages Dataset","input_info":'Enter a college-related message as text. The model analyzes the words in the message.',"output_info":'Predicted message category: Academic, Promotional or Placement, according to the CSV labels.',"kind":"text"},
 {"id":"svm_linear","no":"09","name":"Linear SVM","category":"Classification","desc":"Classify banknotes as genuine or counterfeit.","dataset":"data_banknote_authentication.csv — Banknote Authentication Dataset","input_info":'Enter variance, skewness, curtosis and entropy measurements extracted from a banknote image.',"output_info":'Predicted banknote class: Genuine (0) or Counterfeit (1).',"kind":"classification"},
 {"id":"svm_rbf","no":"10","name":"Non-Linear SVM","category":"Classification","desc":"Classify banknotes using an RBF kernel.","dataset":"data_banknote_authentication.csv — Banknote Authentication Dataset","input_info":'Enter variance, skewness, curtosis and entropy measurements extracted from a banknote image.',"output_info":'Predicted banknote class: Genuine (0) or Counterfeit (1), using the RBF kernel.',"kind":"classification"},
 {"id":"rf","no":"11","name":"Random Forest","category":"Classification","desc":"Predict diabetes outcome using the Random Forest lab dataset.","dataset":"diabetes.csv — Pima Indians Diabetes Dataset","input_info":'Enter Pregnancies, Glucose, BloodPressure, BMI and Age. Other features use dataset medians.',"output_info":'Predicted diabetes outcome: diabetes detected (1) or not detected (0).',"kind":"classification"},
]

def csv(name):
    return pd.read_csv(os.path.join(DATA, name))

def load_experiment(key):
    if key == "slr":
        df = csv("placement.csv")
        return df[["cgpa"]], df["package"], "regression", ["cgpa"], "Package"
    if key == "mlr":
        df = csv("advertising.csv")
        return df[["TV", "Newspaper"]], df["Sales"], "regression", ["TV", "Newspaper"], "Sales"
    if key == "logbin":
        df = csv("placement_binary.csv").drop(columns=["Unnamed: 0"], errors="ignore")
        return df[["cgpa", "iq"]], df["placement"], "classification", ["cgpa", "iq"], "Placement"
    if key == "logmulti":
        iris = load_iris(as_frame=True)
        return iris.data, iris.target, "classification", list(iris.data.columns), "Species"
    if key == "tree":
        df = csv("heart_data.csv")
        features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        return df.drop(columns=["target"]), df["target"], "classification", features, "Heart disease"
    if key == "knn":
        df = csv("diabetes_knn.csv")
        features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
        return df.drop(columns=["Outcome"]), df["Outcome"], "classification", features, "Diabetes"
    if key in ("svm_linear", "svm_rbf"):
        df = csv("banknote.csv")
        features = ["variance", "skewness", "curtosis", "entropy"]
        return df[features], df["class"], "classification", features, "Banknote"
    if key == "rf":
        df = csv("diabetes_rf.csv")
        features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
        return df.drop(columns=["Outcome"]), df["Outcome"], "classification", features, "Diabetes"
    if key == "dataset":
        df = csv("student_dataset.csv")
        return df.drop(columns=["Placed"]), df["Placed"], "analysis", list(df.columns[:-1]), "Student dataset"
    if key == "nb":
        df = csv("college_messages.csv")
        return df["message"], df["label"], "text", ["message"], "Message label"
    raise ValueError("Unknown experiment")

def fit_classifier(key, X, y):
    if key == "logbin":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    if key == "logmulti":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    if key == "tree":
        return DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    if key == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if key == "svm_linear":
        return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    if key == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    if key == "rf":
        return RandomForestClassifier(n_estimators=150, random_state=42)
    raise ValueError("Unknown classifier")

def evaluate_classifier(model, X, y):
    stratify = y if y.nunique() > 1 else None
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    model.fit(Xt, yt)
    pred = model.predict(Xv)
    return [
        ("Accuracy", f"{accuracy_score(yv,pred):.3f}"),
        ("Precision (weighted)", f"{precision_score(yv,pred,average='weighted',zero_division=0):.3f}"),
        ("Recall (weighted)", f"{recall_score(yv,pred,average='weighted',zero_division=0):.3f}"),
        ("F1-score (weighted)", f"{f1_score(yv,pred,average='weighted',zero_division=0):.3f}")
    ]


ALGORITHM_INFO = {
    "dataset": ("Dataset Exploration", "Explore the structure, sample records, missing values and statistical summaries of a dataset. No predictive model is trained in this experiment."),
    "slr": ("Simple Linear Regression", "A regression method that learns a straight-line relationship between one input feature (CGPA) and a continuous target (package)."),
    "mlr": ("Multiple Linear Regression", "A regression method that uses more than one input feature (TV and Newspaper advertising budgets) to estimate a continuous sales value."),
    "logbin": ("Binary Logistic Regression", "A classification algorithm that estimates the probability of one of two placement outcomes: Placed or Not placed."),
    "logmulti": ("Multiclass Logistic Regression", "A classification method that estimates class probabilities and selects one of the three Iris species."),
    "tree": ("Decision Tree", "A tree-based classifier that makes predictions by following feature-based decision rules from the root to a leaf."),
    "knn": ("K-Nearest Neighbors", "A distance-based classifier that predicts the class most common among the five nearest training examples."),
    "nb": ("Multinomial Naive Bayes", "A probabilistic text classifier that uses word counts and Bayes' theorem to assign a message category."),
    "svm_linear": ("Linear Support Vector Machine", "A support vector classifier that learns a linear separating boundary between the two banknote classes."),
    "svm_rbf": ("Non-Linear Support Vector Machine (RBF)", "An SVM using the Radial Basis Function kernel to model a non-linear boundary between banknote classes."),
    "rf": ("Random Forest", "An ensemble classifier that combines predictions from many decision trees to improve stability and predictive performance.")
}

def chart_image(fig):
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def regression_chart(actual, predicted):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.scatter(actual, predicted, alpha=0.75)
    low = min(float(np.min(actual)), float(np.min(predicted)))
    high = max(float(np.max(actual)), float(np.max(predicted)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title("Actual vs Predicted Values")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return chart_image(fig)

def classification_charts(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    return chart_image(fig)

def importance_chart(model, feature_names):
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    if not hasattr(estimator, "feature_importances_"):
        return None
    values = estimator.feature_importances_
    order = np.argsort(values)[-10:]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.barh(np.array(feature_names)[order], values[order])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    return chart_image(fig)

@app.route("/")
def home():
    regression_count = sum(e["kind"] == "regression" for e in EXPERIMENTS)
    classification_count = sum(e["kind"] == "classification" for e in EXPERIMENTS)
    other_count = len(EXPERIMENTS) - regression_count - classification_count
    return render_template(
        "home.html",
        experiments=EXPERIMENTS,
        total_experiments=len(EXPERIMENTS),
        regression_count=regression_count,
        classification_count=classification_count,
        other_count=other_count,
        recent_activity=RECENT_ACTIVITY
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/experiment/<key>", methods=["GET","POST"])
def experiment(key):
    exp = next((e for e in EXPERIMENTS if e["id"] == key), None)
    if exp is None:
        return "Experiment not found", 404
    X, y, task, selected_features, target_name = load_experiment(key)
    result = None
    defaults = {}
    field_bounds = {}
    if task == "text":
        defaults["message"] = ""
    elif task != "analysis":
        for f in selected_features:
            defaults[f] = round(float(X[f].median()), 3)
            field_bounds[f] = (float(X[f].min()), float(X[f].max()))
    if request.method == "POST":
        if task not in ("analysis", "text"):
            invalid_values = []
            for field in selected_features:
                try:
                    entered = float(request.form.get(field, ""))
                    low, high = field_bounds[field]
                    if not np.isfinite(entered):
                        invalid_values.append(f"{field} must be a finite number.")
                    elif entered < low or entered > high:
                        invalid_values.append(f"{field} should be between {low:.3g} and {high:.3g} (the range in this dataset).")
                except (TypeError, ValueError):
                    invalid_values.append(f"Enter a valid number for {field}.")
            if invalid_values:
                result = {"error": " ".join(invalid_values)}
        if request.method == "POST" and result is None and task == "analysis":
            df = csv("student_dataset.csv")
            result = {
                "message": "Dataset inspection completed using the sample student dataset from the Experiment 01 notebook.",
                "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
                "preview": df.head(8).to_html(classes="data-table", index=False, border=0),
                "summary": df.describe(include="all").to_html(classes="data-table", border=0)
            }
        elif result is None and task == "regression":
            model = LinearRegression().fit(X, y)
            vals = [float(request.form.get(f, defaults[f])) for f in selected_features]
            pred_value = float(model.predict(pd.DataFrame([vals], columns=selected_features))[0])
            Xt, Xv, yt, yv = train_test_split(X, y, test_size=.2, random_state=42)
            ev = LinearRegression().fit(Xt, yt)
            pv = ev.predict(Xv)
            result = {"prediction": f"{pred_value:.3f}", "message": f"Predicted {target_name}",
                      "metrics": [("R² score", f"{r2_score(yv,pv):.3f}"),
                                  ("Mean Absolute Error", f"{mean_absolute_error(yv,pv):.3f}"),
                                  ("RMSE", f"{np.sqrt(mean_squared_error(yv,pv)):.3f}")],
                      "chart": regression_chart(yv, pv)}
        elif result is None and task == "text":
            message = request.form.get("message", "").strip()
            if not message:
                result = {"error": "Please enter a message to classify."}
            else:
                Xt, Xv, yt, yv = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
                vectorizer = CountVectorizer()
                Xt_vec = vectorizer.fit_transform(Xt)
                Xv_vec = vectorizer.transform(Xv)
                model = MultinomialNB().fit(Xt_vec, yt)
                prediction = model.predict(vectorizer.transform([message]))[0]
                metrics_pred = model.predict(Xv_vec)
                result = {"prediction": str(prediction), "message": "Predicted message category",
                          "metrics": [("Accuracy", f"{accuracy_score(yv,metrics_pred):.3f}")]}
        elif result is None:
            model = fit_classifier(key, X, y)
            metrics = evaluate_classifier(model, X, y)
            # Evaluate separately to create a confusion matrix for classification.
            stratify_y = y if y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y
            )
            eval_model = fit_classifier(key, X_train, y_train)
            eval_model.fit(X_train, y_train)
            eval_predictions = eval_model.predict(X_test)
            class_names = None
            if key in ("svm_linear", "svm_rbf"):
                class_names = ["Genuine", "Counterfeit"]
            elif key in ("knn", "rf"):
                class_names = ["No diabetes", "Diabetes"]
            elif key == "tree":
                class_names = ["No disease", "Disease"]
            elif key == "logbin":
                class_names = ["Not placed", "Placed"]
            elif key == "logmulti":
                class_names = list(load_iris().target_names)
            cm_chart = classification_charts(y_test, eval_predictions, labels=class_names)
            all_values = {f: float(X[f].median()) for f in X.columns}
            for f in selected_features:
                try:
                    all_values[f] = float(request.form.get(f, defaults[f]))
                except ValueError:
                    all_values[f] = defaults[f]
            input_df = pd.DataFrame([[all_values[f] for f in X.columns]], columns=X.columns)
            model.fit(X, y)
            prediction = model.predict(input_df)[0]
            if key in ("svm_linear", "svm_rbf"):
                prediction_label = "Genuine" if int(prediction) == 0 else "Counterfeit"
            elif key in ("knn", "rf"):
                prediction_label = "Diabetes detected" if int(prediction) == 1 else "No diabetes detected"
            elif key == "tree":
                prediction_label = "Heart disease detected" if int(prediction) == 1 else "No heart disease detected"
            elif key == "logbin":
                prediction_label = "Placed" if int(prediction) == 1 else "Not placed"
            elif key == "logmulti":
                prediction_label = load_iris().target_names[int(prediction)]
            else:
                prediction_label = str(prediction)
            result = {"prediction": prediction_label, "message": "Prediction result",
                      "metrics": metrics, "chart": cm_chart}
            if key in ("tree", "rf"):
                result["importance_chart"] = importance_chart(model, list(X.columns))
    # Add successful predictions/analyses to the dashboard's recent activity list.
    if request.method == "POST" and result and not result.get("error"):
        activity_output = result.get("prediction") or result.get("shape") or "Analysis completed"
        RECENT_ACTIVITY.insert(0, {
            "experiment": exp["name"],
            "output": str(activity_output),
            "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
        })
        del RECENT_ACTIVITY[5:]

    algorithm_name, algorithm_description = ALGORITHM_INFO[key]
    return render_template("experiment.html", exp=exp, features=selected_features, defaults=defaults,
                           field_bounds=field_bounds, result=result,
                           hidden_count=(len(X.columns)-len(selected_features)) if task != "text" else 0,
                           task=task, algorithm_name=algorithm_name,
                           algorithm_description=algorithm_description)

if __name__ == "__main__":
    app.run(debug=True)
