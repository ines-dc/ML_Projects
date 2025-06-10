import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("improved_disease_dataset.csv")

encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

plt.figure(figsize=(18,8))
sns.countplot(x=y)
plt.title("Disease Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.show()

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("Resampled Class Distribution:\n", pd.Series(y_resampled).value_counts())

from statistics import mean
from sklearn.model_selection import StratifiedKFold


if "gender" in X_resampled.columns:
    le = LabelEncoder()
    X_resampled["gender"] = le.fit_transform(X_resampled["gender"])

X_resampled = X_resampled.fillna(0)

if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

models={
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

cv_scoring = "accuracy"
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    try:
        scores = cross_val_score(
            model,
            X_resampled,
            y_resampled,
            cv=stratified_kfold,
            scoring=cv_scoring,
            n_jobs=-1,
            error_score= "raise"
        )
        print("="*50)
        print(f"Model: {model_name}")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {mean(scores):.4f}")
    except Exception as e:
        print("="*50)
        print((f"Model: {model_name} failed with error: {e}"))


svm_model = SVC()
svm_model.fit(X_resampled,y_resampled)
svm_preds = svm_model.predict(X_resampled)

cf_matrix_svm = confusion_matrix(y_resampled,svm_preds)
plt.figure(figsize= (12,8))
sns.heatmap(data=cf_matrix_svm, annot=True, fmt = "d")
plt.title("Confusion Matrix for SVM Classifier")
plt.show()
print("--"*50)
print(f"SVM Accuracy: {accuracy_score(y_resampled, svm_preds) *100:.2f}%")

nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)
nb_preds = nb_model.predict(X_resampled)

cf_matrix_nb = confusion_matrix(y_resampled, nb_preds)
plt.figure(figsize= (12,8))
sns.heatmap(data=cf_matrix_nb, annot=True, fmt = "d")
plt.title("Confusion Matrix for Naive Bayes Classifier")
plt.show()
print("--"*50)
print(f"Naive Bayes Accuracy: {accuracy_score(y_resampled, nb_preds) *100:.2f}%")


rf_model = RandomForestClassifier()
rf_model.fit(X_resampled, y_resampled)
rf_preds = rf_model.predict(X_resampled)
cf_matrix_rf = confusion_matrix(y_resampled, rf_preds)
plt.figure(figsize= (12,8))
sns.heatmap(data=cf_matrix_rf, annot=True, fmt = "d")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()
print("--"*50)
print(f"Random Forrest Accuracy: {accuracy_score(y_resampled, rf_preds) *100:.2f}%")

from statistics import mode

final_preds = [mode([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

cf_matrix_combined = confusion_matrix(y_resampled, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_combined, annot=True, fmt="d")
plt.title("Confusion Matrix for Combined Model")
plt.show()
print("--"*50)
print(f"Combined Model Accuracy: {accuracy_score(y_resampled, final_preds) * 100:.2f}%")

symptoms = X.columns.values
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}


def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)

    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)
    input_df = pd.DataFrame(input_data, columns=X.columns)

    rf_pred = encoder.classes_[rf_model.predict(input_df)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_df)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_df)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])

    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }

print("--"*50)
print(predict_disease("Itching,Skin Rash,Nodal Skin Eruptions"))