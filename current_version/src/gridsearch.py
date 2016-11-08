from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster

def gridsearch_launch(gridSamples, gridTarget, nb_clusters):

    # print("SAMPLES")
    # # print(gridSamples);
    # print(gridTarget);
    #
    # print(len(gridTarget))
    # print(len(gridSamples))

    X = gridSamples.reshape((len(gridSamples), -1))
    y = gridTarget

    # Split the dataset


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # tuned_parameters = [
    #     {'init': ['k-means++'], 'n_clusters': [3, 4, 5, 6, 7], 'n_init': [5, 10, 20, 30, 40], 'algorithm': ['auto', 'full', 'elkan']},
    #     {'init': ['random'], 'n_clusters': [3, 4, 5, 6, 7], 'n_init': [5, 10, 20, 30, 40], 'algorithm': ['auto', 'full', 'elkan']},
    # ]

    tuned_parameters = [
        {'init': ['k-means++'], 'n_init': [5, 10, 20, 30, 40], 'algorithm': ['auto', 'full', 'elkan']},
        {'init': ['random'], 'n_init': [5, 10, 20, 30, 40], 'algorithm': ['auto', 'full', 'elkan']},
    ]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(cluster.KMeans(n_clusters=nb_clusters), tuned_parameters, cv=5,
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
