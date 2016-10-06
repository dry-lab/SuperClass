import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


def k_mean_launcher(samples,pits, N_CLUSTERS):

        k_means = KMeans(init='k-means++', n_clusters=N_CLUSTERS, n_init=50)
        k_means.fit(samples)
        k_means_labels = k_means.labels_
        k_means_cluster_centers = k_means.cluster_centers_

        classif = k_means.predict(samples)
        classif_df = pd.DataFrame(classif, index=np.arange(1, len(classif) + 1))
        classif_df.columns = ['scores']

        pit_df = pd.DataFrame(pits, index=np.arange(1, len(pits) + 1))
        pit_df.columns = ['pits']

        result_per_pit = pd.concat([classif_df, pit_df], axis=1, verify_integrity=False)
        result_per_pit = result_per_pit[['scores', 'pits']].values
        result_per_pit_df = pd.DataFrame(result_per_pit, index=np.arange(len(result_per_pit)))
        result_per_pit_df.columns = ['scores', 'pits']

        grouped = result_per_pit_df.groupby('pits')

        print("Clustering sparse data with %s" % k_means)

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(pits, k_means.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(pits, k_means.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(pits, k_means.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(pits, k_means.labels_))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(samples, k_means.labels_, sample_size=1000))
        print()

        for pit, cluster in grouped:
            print cluster
        print result_per_pit_df

        fig = plt.figure(figsize=(8, 3))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#4E9A06']  # ,'#555555']

        # Plot
        ax = fig.add_subplot(1, 3, 1)
        for k, col in zip(range(N_CLUSTERS), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            ax.plot(samples[my_members, 0], samples[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)
        ax.set_title('KMeans')
        ax.set_xticks(())
        ax.set_yticks(())

        plt.show()