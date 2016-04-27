# -*- coding: utf-8 -*-

"""Try the classification stuff

"""
import warnings
import logging
logging.basicConfig(level=logging.DEBUG)

# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import additive_chi2_kernel
from sklearn.decomposition import PCA

from constants import *
# code



def inline_merge_pits_in_conditions(samples, groups, key='position'):
    """Merge several pits together in order to be able to says that such group of pits
    correspond to the same kind of experiment
    """

    samples[key] = None

    #print ("KEY : ", key)
    for pit, cond in groups.items():
        #print ("samples[key] : ", samples[key])
        samples[key][samples['pit'] == pit] = cond

    assert (samples[key] != -1).all(), "Some samples have no condition"





class ClassificationExperiment(object):
    """Manage the classification experiment"""

    def __init__(self, data_to_copy, label):
        """Initialize the various objects.
        
        data_to_copy:
            original data object we want to copy in order to modify it
        label:
            Label of the experiment
        """


        self._data = pd.DataFrame(data_to_copy, copy=True)
        self._label = label



    def get_columns_to_remove(self):
        """Get the list of columns to remove from the dataset"""
        return []

    def _remove_unused_columns(self):
        """Remove the columns not intersing expect if they are pit, position or condition"""
        for column in self.get_columns_to_remove():
            if column not in ['pit', 'position', 'condition']:
                del self._data[column]


    def _compute_comparison_scores(self, key='condition', ):
        """Try to compare the samples.
        XXX Need to improve that when we will have more data.
        XXX Allow to change the comparison method
        XXX we do not assume that the distance measure is symetric, so 
        the comparison of r1 to r2 and r2 to r2 can give different scores
        """
        samples = self._data

        grouped = samples.groupby(key)
        intra_scores = []
        inter_scores = []

        # We do not want columns other than the ones of extracted features
        columns = samples.columns.tolist()
        for column in ['pit', 'position', 'condition']:
            del columns[columns.index(column)]

        for name, group in grouped:
            logging.info('Compute distance within group ' + name)


            # Compute intra scores
            for idx1, row1 in group.iterrows():
                for idx2, row2 in group.iterrows():
                    # do not compare identic rows
                    if idx1 == idx2:
                        continue
                    else:
                        logging.debug('\tCompare %s to %s' % (idx1, idx2))
                        intra_scores.append(self.compare_samples(row1[columns], row2[columns]))

        for name1, group1 in grouped:
            for name2, group2 in grouped:
                if name1 == name2:
                    continue
                else:
                    logging.info("Compute distance between " + name + " and " + name2)
                    for idx1, row1 in group1.iterrows():
                        for idx2, row2 in group2.iterrows():
                            logging.debug('\tCompare %s to %s' % (idx1, idx2))
                            inter_scores.append(self.compare_samples(row1[columns], row2[columns]))
        

        self._intra = intra_scores
        self._inter = inter_scores

    # Unused ?
    # def chi2_distance(self, sample1, sample2):
    #     """Compare two samples together using the Chi2 method"""

    #     return ((sample1 - sample2)*(sample1 - sample2) / (sample1+sample2 + 0.0001)).sum()

    # def euclidean_distance(self, sample1, sample2):
    #     """Compare two samples using the euclidean distance"""
    #     return ((sample1-sample2)*(sample1-sample2)**2).sum()

    # def visualize_samples(self, key='condition'):
    #     """Visualize the samples depending on the experimental conditions"""
    #     samples = self._data
    #     samples.groupby(key).boxplot(rot=90, grid=True)
    #     plt.savefig('data_distribution.pdf')


    def visualize_performance(self):
        intra = self._intra
        inter = self._inter

        labels = [1]*len(intra) + [-1]*len(inter)
        scores = intra+inter

        self._common_visualize_performance( labels, scores)

        plt.figure()
        plt.boxplot([intra, inter])
        plt.xticks([1, 2], ['intra', 'inter'])
        plt.title('Distribution of scores')
        plt.savefig('comparison_score_distribution.pdf')


        plt.figure()
        start = np.min(np.min(intra), np.min(inter))
        end = np.max(np.max(intra), np.max(inter))
        intra_hist, intra_bin = np.histogram(intra,50, (start, end))
        inter_hist, inter_bin = np.histogram(inter,50, (start, end))


        plt.plot(intra_bin[:-1], intra_hist/float(intra_hist.sum()), label='intra', color='blue')
        plt.plot(inter_bin[:-1], inter_hist/float(inter_hist.sum()), label='inter', color='red')
        plt.legend()
        plt.xlabel('Comparison scores')
        plt.ylabel('Probability')
        plt.title('Score distribution')



    def _common_visualize_performance(self, labels, scores):
        plt.figure()
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive (A) Rate')
        plt.ylabel('True Positive (A) Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.pdf')

    def run(self):
        self._remove_unused_columns()
        self._compute_comparison_scores()


class ClassificationExperimentMachineLearning(ClassificationExperiment):
    """Instead of using distance functions, use machine learning (ie, need to compute a model)"""

    def __init__(self, data, label, classifier):
        assert classifier in ('SVC', 'RF', 'SVC-PCA', 'RF-PCA', 'KNN_EUCL', 'KNN_CHI2')
        super(ClassificationExperimentMachineLearning, self).__init__(data, label)
        self._selected_classifier = classifier

    def run(self):
        self._remove_unused_columns()
        self._compute_comparison_scoress_with_cross_validation()


    def _compute_comparison_scoress_with_cross_validation(self, key='condition'):
        """Compute the scores, thanks to cross validation"""


        samples = self._data
        labels = samples[key].values

        print"########### samples"
        print samples


        # make a transformation of the labels
        unique_labels = np.unique(labels)

        if len(unique_labels) == 2:
            assert len(unique_labels) == 2, 'We have not tested something else' #XXX  Need to modify the scripts for something else
            assert 'LIVING' in unique_labels
            assert 'FIXED' in unique_labels
            labels[labels=='LIVING'] = +1
            labels[labels=='FIXED'] = -1
        else:
            warnings.warn("We use several labels and it has not been deeply tested")
            print labels
        # get the columns of interest
        columns = samples.columns.tolist()
        print"########### columns"
        print columns
        for column in ['pit', 'position', 'condition']:
            if column in columns:
                del columns[columns.index(column)] 

        # We have few elements => use a leave one out method
        values = samples[columns].values

        print"########### LABELS"
        print labels
        print"########### values"
        print values

        scores = np.zeros(len(values))
        loo = LeaveOneOut(len(values))
        logging.info('Number of samples ' + str(len(values)))


        for train, test in loo:
            # Get the partitions
            samples_train = values[train]
            # print"########### SAMPLES TRAIN"
            # print samples_train

            samples_test = values[test]
            # print"########### SAMPLES TEST"
            # print samples_test

            labels_train = labels[train]
            # print"########### labels train"
            # print labels_train

            labels_test = labels[test]
            # print"########### labels TEST"
            # print labels_test


            # Normalize data when KNN is not used (ie. metric with histo comparison)
            if self._selected_classifier not in ('KNN_CHI2', 'KNN_EUCL'):
                scaler = StandardScaler().fit(samples_train)
                samples_train = scaler.transform(samples_train)
                samples_test = scaler.transform(samples_test)

            if self._selected_classifier in ('SVC-PCA', 'RF-PCA'):
                decomposition = PCA().fit(samples_train)
                samples_train = decomposition.transform(samples_train)
                samples_test = decomposition.transform(samples_test)

            # Train the classifier
            if self._selected_classifier.startswith('SVC'):
                clf = SVC(probability=True)
            elif self._selected_classifier.startswith('RF'):
                clf = RandomForestClassifier(n_jobs=-1)
            elif self._selected_classifier.startswith('KNN_EUCL'):
                clf = KNeighborsClassifier(metric='minkowski', p=2)
            elif self._selected_classifier.startswith('KNN_CHI2'):
                clf = KNeighborsClassifier(metric=additive_chi2_kernel)


            clf.fit(samples_train, labels_train.tolist())

            # make the classification
            # XXX Hack to see if it is multiclass or not
            if 2 == len(np.unique(labels_train)) :
                score = clf.predict_proba(samples_test)[:,1]
            else:
                score = clf.predict(samples_test)
            scores[test] = score

        self._scores = scores
        self._true_labels = labels
        # print "SCORES : "
        # print scores
        # print "LABELS :"
        # print labels
        # print "#########"


    def visualize_performance(self):

        self._common_visualize_performance(self._true_labels, self._scores)



class DensityHistoClassifier(ClassificationExperimentMachineLearning):
    """We want an experiment which keeps only the density histogram"""

    def __init__(self, data, classifier, columns):
        super(DensityHistoClassifier, self).__init__(data, "density_histo_classifier", classifier)
        self.columns = columns

    def get_columns_to_remove(self):
        columns = self._data.columns.tolist()


        if self.columns == 'density_hist':
            keep_only = DENSITY_HISTOGRAM_LABELS 
        elif self.columns == 'trajectory_hist':
            keep_only = TRAJECTORY_FRAMES_HISTOGRAM_LABELS
        elif self.columns == 'msd_hist':
            keep_only = MSD_HISTOGRAM_LABELS
        elif self.columns == 'all':
            return []
        else:
            raise Exception(self.columns + ' unknown')


        return set(columns) - set(keep_only)

    


        

# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2013, LaBRI'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'romain.giot@u-bordeaux1.fr'
__status__ = 'Prototype'

