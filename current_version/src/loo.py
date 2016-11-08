# imports

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import additive_chi2_kernel
from sklearn.decomposition import PCA


def launcher_loo(key='condition', samples, labels, selected_classifier):
    """Compute the scores, thanks to cross validation"""

    samples2=samples
    # make a transformation of the labels
    labels = samples[key].values
    unique_labels = np.unique(labels)

    # if len(unique_labels) == 2:
    #     assert len(unique_labels) == 2, 'We have not tested something else' #XXX  Need to modify the scripts for something else
    #     assert 'LIVING' in unique_labels
    #     assert 'FIXED' in unique_labels
    #     labels[labels=='LIVING'] = +1
    #     labels[labels=='FIXED'] = -1
    # else:
    #     #warnings.warn("We use several labels and it has not been deeply tested")
    #     labels[labels=='condition1'] = 1
    #     labels[labels=='condition2'] = 2
    #     labels[labels=='condition3'] = 3
    #     labels[labels=='condition4'] = 4

    # get the columns of interest
    columns = samples.columns.tolist()

    # We have few elements => use a leave one out method
    values = samples[columns].values
    scores = np.zeros(len(values))
    scores_label=np.zeros(len(values))
    loo = LeaveOneOut(len(values))

    for train, test in loo:

        # Get the partitions
        samples_train = values[train]
        samples_test = values[test].reshape(1,  - 1)
        print samples_test
        labels_train = labels[train]
        labels_test = labels[test]

        # Normalize data when KNN is not used (ie. metric with histo comparison)
        if selected_classifier not in ('KNN_CHI2', 'KNN_EUCL'):
            scaler = StandardScaler().fit(samples_train)
            samples_train = scaler.transform(samples_train)
            samples_test = scaler.transform(samples_test)

        if self._selected_classifier in ('SVC-PCA', 'RF-PCA'):
            decomposition = PCA().fit(samples_train)
            samples_train = decomposition.transform(samples_train)
            samples_test = decomposition.transform(samples_test)

        # Train the classifier
        if selected_classifier.startswith('SVC'):
            clf = SVC(probability=True)
            clf.fit(samples_train, labels_train.tolist())

        elif selected_classifier.startswith('RF'):
            clf = RandomForestClassifier(n_jobs=-1)
            clf.fit(samples_train, labels_train.tolist())

        elif selected_classifier.startswith('KNN_EUCL'):
            clf = KNeighborsClassifier(metric='minkowski', p=2)
            clf.fit(samples_train, labels_train.tolist())

        elif selected_classifier.startswith('KNN_CHI2'):
            logging.info('selected classifier KNN_CHI2')
            clf = KNeighborsClassifier(metric=additive_chi2_kernel)
            clf.fit(samples_train, labels_train.tolist())

        # make the classification
        # XXX Hack to see if it is multiclass or not
        if 2 == len(np.unique(labels_train)) :
            score = clf.predict_proba(samples_test)[:,-1]
            #print clf.predict(samples_test)[:,-1]
            score_label = clf.predict(samples_test)
        else:
            score = clf.predict_proba(samples_test)
            score_label = clf.predict(samples_test)

        scores[test] = score
        scores_label[test] = score_label

    pits=samples2['pit']
    classif_df=pd.DataFrame(scores_label,index=np.arange(1,len(scores_label)+1))
    classif_df.columns = ['scores']

    pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
    pit_df.columns = ['pits']

    result_per_pit = pd.concat([classif_df, pit_df], axis=1,verify_integrity=False)

    result_per_pit= result_per_pit[['scores','pits']].values
    result_per_pit_df=pd.DataFrame(result_per_pit,index=np.arange(len(result_per_pit)))
    result_per_pit_df.columns =['scores','pits']
    #result_per_pit_df = result_per_pit_df.set_index('index')

    grouped = result_per_pit_df.groupby('pits')

    for pit,cluster in grouped:
        print cluster
    #print scores
    #print scores_label
