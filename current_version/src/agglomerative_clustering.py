import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from sklearn.cluster import AgglomerativeClustering



def agglomerative_clustering_launcher(samples, pits, type, N_CLUSTERS):
    knn_graph = kneighbors_graph(samples, 3, include_self=False)

    for connectivity in (None, knn_graph):
        for n_clusters in (3, 3):
            plt.figure(figsize=(10, 4))
            for index, linkage in enumerate(('average', 'complete', 'ward')):
                plt.subplot(1, 3, index + 1)
                model = AgglomerativeClustering(linkage=linkage,
                                                connectivity=connectivity,
                                                n_clusters=n_clusters)
                t0 = time.time()
                model.fit(samples)
                elapsed_time = time.time() - t0
                plt.scatter(samples[:, 0], samples[:, 1], c=model.labels_,
                            cmap=plt.cm.spectral)
                plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
                          fontdict=dict(verticalalignment='top'))
                plt.axis('equal')
                plt.axis('off')

                plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                    left=0, right=1)
                plt.suptitle('n_cluster=%i, connectivity=%r' %
                             (n_clusters, connectivity is not None), size=17)

    plt.show()