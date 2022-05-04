import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

# Using kmeans as clustering to cluster universities into two groups, public & private
# Although the dataset has labels, we will ignore them since KMeans Clustering is an unsupervised learning algorithm

desired_width = 410
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

df = pd.read_csv('College_Data', index_col=0)
print(df.head())

# Preliminary data visualizations
sns.scatterplot(x=df['Room.Board'], y=df['Grad.Rate'], hue=df['Private'])
sns.scatterplot(x=df['Outstate'], y=df['F.Undergrad'], hue=df['Private'])
# plt.show()

# KMeans Clustering implementation
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private', axis=1))
print(kmeans.cluster_centers_)


# Utilizing the labels from dataset to compare the performance of our model vs actual dataset
def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0


# Creating a new column, 'Cluster', in original dataframe where 1 = 'Private' and 0 = 'Public'
df['Cluster'] = df['Private'].apply(converter)

# Observing accuracy
print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))
