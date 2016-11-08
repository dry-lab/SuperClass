from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster

def gridsearch_various_launch(gridSamples, gridTarget, nb_clusters):

    X = gridSamples.reshape((len(gridSamples), -1))
    y = gridTarget

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    tuned_parameters_kmeans = [
        {'init': ['k-means++'], 'n_init': [5, 10, 20, 30, 40], 'algorithm': ['auto', 'full', 'elkan']},
        {'init': ['random'], 'n_init': [5, 10, 20, 30, 40], 'algorithm': ['auto', 'full', 'elkan']},
    ]

    tuned_parameters_two_means = [
        {'init': ['k-means++'], 'n_init': [5, 10, 20, 30, 40], 'algorithm': ['auto', 'full', 'elkan']},
        {'init': ['random'], 'n_init': [5, 10, 20, 30, 40], 'algorithm': ['auto', 'full', 'elkan']},
    ]

    tuned_parameters_affinity =[
        {'affinity':['euclidean'], 'damping':[0.3,0.4,0.5,0.6,0.7,0.8,0.9], 'preference':[-200,-150,-100,-50,0,50,100,150,200]},
        {'affinity': ['precomputed'], 'damping': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'preference': [-200, -150, -100, -50, 0, 50, 100, 150, 200]}
    ]

    tuned_parameters_ms =[
        {'bandwidth':[], 'bin_seeding':[True, False]}
    ]
    tuned_parameters_spectral=[
        {'eigen_solver':['arpack', 'lobpcg', 'amg'], 'affinity':['nearest_neighbors', 'precomputed', 'rbf'], 'n_init':[5,7,10,12,15]}
    ]

    tuned_parameters_wrad =[
        {'linkage':['ward', 'complete', 'average'], 'affinity':['euclidean','l1', 'l2', 'manhattan', 'cosine', 'precomputed'] }
    ]


    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=5, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)


        # create clustering estimators
        k_means = cluster.KMeans(init='k-means++', n_clusters=nb_clusters, n_init=20, algorithm='full')
        two_means = cluster.MiniBatchKMeans(n_clusters=nb_clusters)
        affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200, affinity='euclidean')
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        spectral = cluster.SpectralClustering(n_clusters=nb_clusters, eigen_solver='arpack',affinity="nearest_neighbors")
        ward = cluster.AgglomerativeClustering(n_clusters=nb_clusters, linkage='ward', connectivity=connectivity)
        average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=nb_clusters, connectivity=connectivity)
        dbscan = cluster.DBSCAN(eps=.3)
        birch = cluster.Birch(n_clusters=nb_clusters)



        clustering_algorithms = [
            k_means, two_means, affinity_propagation, ms, spectral, ward, average_linkage,
            dbscan, birch]



        clf = GridSearchCV(cluster.KMeans(n_clusters=nb_clusters), tuned_parameters_kmeans, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
