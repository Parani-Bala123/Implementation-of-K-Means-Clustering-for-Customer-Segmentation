# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset using pandas, inspect its structure (head(), info()), and check for missing values to ensure data quality before applying clustering
2. Use a loop to run K-Means for cluster numbers 1 to 10, calculate the wcss for each, and plot them. The “elbow point” in the plot helps decide the ideal number of clusters (here, 5).
3. Fit the K-Means model with the optimal number of clusters (5) on selected features (Annual Income and Spending Score), and predict cluster labels for each customer.
4. Add cluster labels to the dataset and use a scatter plot to visualize different customer segments based on their Annual Income and Spending Score, with each cluster shown in a different color.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Parani Bala M
RegisterNumber: 212224230192
*/

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[] #within cluster sum of square

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow method")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
KMeans(n_clusters=5)

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster1")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="brown",label="cluster2")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="blue",label="cluster4")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="orange",label="cluster5")
plt.legend()
plt.title("Customer segments")
```

## Output:
<img width="595" height="223" alt="image" src="https://github.com/user-attachments/assets/2ef6fb5c-38bf-4209-a23d-9fa4daa55129" />
<img width="602" height="262" alt="image" src="https://github.com/user-attachments/assets/ed0ab773-681d-476e-82cf-acc36e69abd7" />
<img width="545" height="160" alt="image" src="https://github.com/user-attachments/assets/0e36925a-75f7-4d17-a31d-b70ac2f96f85" />
<img width="863" height="610" alt="image" src="https://github.com/user-attachments/assets/613bce34-2e37-491b-b635-a34f29509848" />
<img width="854" height="96" alt="image" src="https://github.com/user-attachments/assets/2b516d36-7e10-4921-887e-1fe86104bcd7" />
<img width="817" height="227" alt="image" src="https://github.com/user-attachments/assets/9658df59-7424-46bd-a087-1e9f4acaac6f" />
<img width="1032" height="596" alt="image" src="https://github.com/user-attachments/assets/15f1b74d-9217-4d0b-87ef-545e1725bdaa" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
