import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


data = pd.read_csv("creditcard.csv")
print(data.head())
print(data.describe())

#Analyse class distribution
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))

print(f"ratio fraud/valid: {outlierFraction}")
print("Fraud cases : {}".format(len(data[data['Class'] == 1])))
print("Valid cases : {}".format(len(data[data['Class'] == 0])))

#Explore Data transactions Amounts
print("-"*50)
print("Amount details of the fraudulent transaction")
print(fraud.Amount.describe())

print("-"*50)
print("details of valid transaction")
print(valid.Amount.describe())

#Plotting correlation matrix
# correlation_matrix = data.corr()
# print("---correlation_matrix---")
# fig = plt.figure(figsize=(12,9))
# sns.heatmap(correlation_matrix, vmax = 0.8, square = True)
# plt.show()

#Preparing the data set
X = data.drop(["Class"], axis = 1)
Y = data["Class"]
print(f"X shape: {X.shape}")
print(f" Y shape: {Y.shape}")

xData= X.values
yData = Y.values


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size=0.2, random_state=42)

#Building and training the model
from sklearn.ensemble import RandomForestClassifier
rfc = rfc = RandomForestClassifier()
print(">>> Starting Training...")
rfc.fit(xTrain, yTrain)
print(">>> Finished Training...")
print("--"*50)
print(">>> Starting Prediction...")
y_pred = rfc.predict(xTest)
print(">>> Finished Prediction...")
print("--"*50)
print("--"*50)

#Evaluating the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
print(">>> Starting Evaluation...")
accuracy = accuracy_score(yTest, y_pred)
precision = precision_score(yTest, y_pred)
recall = recall_score(yTest, y_pred)
f1 = f1_score(yTest, y_pred)
mcc = matthews_corrcoef(yTest, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

conf_matrix = confusion_matrix(yTest, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()