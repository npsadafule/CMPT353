import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_pca(X):
    """
    Transform data to 2D points for plotting. Should return an array with shape (n, 2).
    """
    # Using MinMaxScaler and PCA in the exact sequence as provided
    flatten_model = make_pipeline(
        MinMaxScaler(),
        PCA(n_components=2)
    )
    X2 = flatten_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)
    return X2

def get_clusters(X):
    """
    Find clusters of the weather data using KMeans.
    """
    model = make_pipeline(
        KMeans(n_clusters=10)
    )
    model.fit(X)
    return model.predict(X)

def main():
    # Reading the data from the provided CSV file
    data = pd.read_csv(sys.argv[1])

    # Extracting the features and city labels
    X = data.drop(columns=['city', 'year']).values
    y = data['city'].values

    # Perform PCA and clustering
    X2 = get_pca(X)
    clusters = get_clusters(X)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='Set1', edgecolor='k', s=30)
    plt.show()

    # Creating a DataFrame to analyze the distribution of cities across clusters
    df = pd.DataFrame({
        'cluster': clusters,
        'city': y
    })
    counts = pd.crosstab(df['city'], df['cluster'])
    print(counts)

if __name__ == '__main__':
    main()
