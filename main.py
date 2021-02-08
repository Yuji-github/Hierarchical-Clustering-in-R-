# Hierarchical Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for feature scaling
from sklearn.preprocessing import StandardScaler

# for splitting data into trains and tests
from sklearn.model_selection import train_test_split

# for dendrogram plot
from scipy.cluster import hierarchy

# for Agglomerative cluster
from sklearn.cluster import AgglomerativeClustering

# plotting 3D
from mpl_toolkits.mplot3d import Axes3D

def HC():
    # import data
    dataset = pd.read_csv('Mall_Customers.csv')

    # encoding Genre
    gender = {'Male': 1, 'Female': 2}
    dataset.Genre = [gender[Item] for Item in dataset.Genre]

    '''
         CustomerID  Genre  Age  Annual Income (k$)  Spending Score (1-100)
    0             1      1   19                  15                      39
    1             2      1   21                  15                      81
    '''

    # independent
    independent = dataset.iloc[:, [2, 3, 4]].values

    # dendrogram plot
    # linkage = Perform hierarchical/agglomerative clustering.
    # more info about linkage:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    # y: nd array = 2d, 3d, 4d any features can handle
    # ward is minimization algorithm.
    dendrogram = hierarchy.dendrogram(hierarchy.linkage(independent, method='ward'))
    plt.title('Dendrogram (Ward)')
    plt.xlabel('Customers')
    plt.ylabel('distance')
    plt.show()

    # train/fit the dataset
    # n_clusters is how many clusters
    # affinity is distance default=’euclidean’
    # linkage The algorithm will merge the pairs of cluster default=’ward’
    hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    dependent = hc.fit_predict(independent)

    print(dependent)

    # 3D plotting
    fig = plt.figure()
    ax = Axes3D(fig)
    item = np.array(dataset['Genre'])
    item2 = np.array(dependent)
    row = 0

    for i, j in zip(item, item2):
        if i == 1:  # male
            if j == 0:
                x1, y1, z1 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax.scatter(xs=x1, ys=y1, zs=z1, marker='o', c='b')
            elif j == 1:
                x2, y2, z2 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax.scatter(xs=x2, ys=y2, zs=z2, marker='o', c='r')
            elif j == 2:
                x3, y3, z3 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax.scatter(xs=x3, ys=y3, zs=z3, marker='o', c='pink')
        elif i == 2:  # female
            if j == 0:
                x1, y1, z1 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax.scatter(xs=x1, ys=y1, zs=z1, marker='x', c='b')
            elif j == 1:
                x2, y2, z2 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax.scatter(xs=x2, ys=y2, zs=z2, marker='x', c='r')
            elif j == 2:
                x3, y3, z3 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax.scatter(xs=x3, ys=y3, zs=z3, marker='x', c='pink')
        row = row + 1

    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    plt.show()


if __name__ == '__main__':
    HC()
