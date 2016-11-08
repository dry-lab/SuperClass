import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster


def various_algorithm_launch(samples, pits, nb_clusters, target):

    np.random.seed(0)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    clustering_names = [
        'Kmeans','MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
        'SpectralClustering', 'Ward', 'AgglomerativeClustering',
        'DBSCAN', 'Birch']


    plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1
    full_result=pd.DataFrame()
    datasets = [samples]

    for i_dataset, dataset in enumerate(datasets):
        
        if i_dataset==len(datasets)-1:
            print i_dataset
            X=samples
            y=pits
        else:
            X, y = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=5, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)


        # create clustering estimators
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=nb_clusters)
        k_means = cluster.KMeans(init='k-means++', n_clusters=nb_clusters, n_init=20, algorithm='full')
        ward = cluster.AgglomerativeClustering(n_clusters=nb_clusters, linkage='ward', connectivity=connectivity)
        spectral = cluster.SpectralClustering(n_clusters=nb_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=0.3, min_samples=10).fit(samples)
        affinity_propagation = cluster.AffinityPropagation(damping = 0.5, max_iter = 200, convergence_iter = 15, copy = True, preference = None, affinity = 'euclidean', verbose = False)
        average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=nb_clusters, connectivity=connectivity)
        birch = cluster.Birch(n_clusters=nb_clusters)


        clustering_algorithms = [
            k_means,two_means, affinity_propagation, ms, spectral, ward, average_linkage,
            dbscan, birch]

        for name, algorithm in zip(clustering_names, clustering_algorithms):
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            classif_df=pd.DataFrame(y_pred,index=np.arange(1,len(y_pred)+1))
            classif_df.columns = [name]
            pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
            pit_df.columns = [target]
            result_per_pit = pd.concat([classif_df, pit_df], axis=1,verify_integrity=False)
            full_result = pd.concat([full_result, result_per_pit[name]], axis=1)


            #plot
            plt.subplot(1, len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
            plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colors[:len(centers)]
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    # print full_result
    # plt.show()
    return full_result