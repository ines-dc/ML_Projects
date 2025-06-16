import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.vq import kmeans

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('new.csv')
print(df.head())
df.shape

df.info()
df.describe().T

for col in df.columns:
    temp = df[col].isnull().sum()
    if temp > 0:
        print(f'column {col} contains {temp} null values')

df = df.dropna()
print("Total values in the dataset after removing the null values:", len(df))

df.nunique()

parts = df["Dt_Customer"].str.split("-", n=3, expand=True)
df["day"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["year"] = parts[2].astype('int')

df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'],
        axis=1,
        inplace=True)

floats, objects = [], []
for col in df.columns:
    if df[col].dtype == object:
        objects.append(col)
    elif df[col].dtype == float:
        floats.append(col)

print(f'objects: {objects}')
print(f'floats: {floats}')

plt.figure(figsize=(15, 10))
for i, col in enumerate(objects):
    plt.subplot(2, 2, i + 1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index[::-1])
    plt.title(f'Inverted Order of {col}')
plt.tight_layout()
plt.show()

print(df["Marital_Status"].value_counts())

plt.subplots(figsize=(15, 10))
for i, col in enumerate(objects):
    plt.subplot(2, 2, i + 1)

    df_melted = df.melt(id_vars=[col], value_vars=['Response'], var_name='hue')
    sns.countplot(x=col, hue='value', data=df_melted)
plt.show()

for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

plt.figure(figsize=(15, 15))
sns.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()

scaler = StandardScaler()
data = scaler.fit_transform(df)

from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
tsne_data= model.fit_transform(df)
plt.figure(figsize=(15,15))
plt.scatter(tsne_data[:,0], tsne_data[:,1])
plt.show()


error = []
for n_clusters in range(1,21):
    model = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=500, random_state=22)
    model.fit(df)
    error.append(model.inertia_)

plt.figure(figsize=(15,15))
sns.lineplot(x=range(1,21), y=error)
sns.scatterplot(x=range(1,21), y=error)
plt.show()

model = KMeans(init='k-means++', n_clusters=6, max_iter=500, random_state=22)
segments = model.fit_predict(df)

plt.figure(figsize=(15,15))
df_tsne = pd.DataFrame({'x': tsne_data[:,0], 'y': tsne_data[:,1], 'segment': segments})
sns.scatterplot(x = 'x', y='y', hue='segment', data=df_tsne, palette='Set2')
plt.show()

df['Segment'] = segments
summary = df.groupby('Segment').mean().round(2)
with open('cluster_summary.txt', 'w') as f:
    f.write(summary.to_string())