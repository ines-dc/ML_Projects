import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

disease_df = pd.read_csv("framingham.csv")

disease_df.head()
print(disease_df.head())

disease_df.drop(["education"], axis=1, inplace=True)
disease_df.rename(columns={"male":"Sex_male"}, inplace=True)
disease_df.dropna(axis=0, inplace=True)
disease_df
print(disease_df)

print(disease_df.TenYearCHD.value_counts())

X = np.asarray(disease_df[["age","Sex_male","cigsPerDay", "totChol", "sysBP", "glucose" ]])
y= np.asarray(disease_df["TenYearCHD"])
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

plt.figure(figsize = (15,15))
sns.countplot(x='TenYearCHD', hue='TenYearCHD', data=disease_df, palette="pastel", legend=False)
plt.show()

laste = disease_df['TenYearCHD'].plot()
plt.show()

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
print('accuracy of the model is:', accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix, classification_report
print("the details for confision matrix is:")
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(
    data = cm,
    columns = ['Predicted:0', "Predicted:1"],
    index = ["Actual:0", "Actual:1"]
)

plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")
plt.show()