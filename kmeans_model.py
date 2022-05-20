from sklearn.cluster import KMeans
import pickle as pk
from pca import reduce_dim
from vector import get_all_vectors
import numpy as np



def train_kmeans():
    X = reduce_dim()
    kmeans = KMeans(n_clusters = 8,n_init = 200).fit(X)
    pk.dump(kmeans, open("kmeans_model.pkl", 'wb')) #Saving the model

    df = get_all_vectors()
    df['cluster'] = kmeans.fit_predict(X)
    # df.to_csv("clusters.csv")

    #print nearest 10 points to each cluster center
    y = df['combined']
    for i in range(8):
        print("For cluster:", i)
        idx = np.argsort(kmeans.transform(X)[:, i])[: 10]
        print(y[idx])

    return df

train_kmeans()
