# -*- coding: utf-8 -*-

"""Tentative to analyse an experiment.


 + Matrix
   + A 1 (=pit)
    + P 1 (=location/position)


"""

# imports
import os
import glob
import logging

import matplotlib.pyplot as plt
from pandas.tools.plotting import andrews_curves


from joblib import Parallel, delayed

from bicproject.readfile import *
from bicproject.classify import *


def _read_samples_per_pit_and_location_inner_loop(position_folder, pit_name):
    """Launche the reading for a position"""

    position_name = os.path.basename(position_folder)
    logging.info("Read position " + position_name + " in pit " + pit_name)

    sample = extract_experiment_features(position_folder) 
    assert sample.shape[0] == 1, "ATTENTION, method tested with a dataset where 'extract_experiment_features' returns one line (maybe it is wrong)"
    sample['pit'] = pit_name
    sample['position'] = position_name


    return sample

def read_samples_per_pit_and_location(data_dir):
    """Read the samples stored in `data_dir`.
    the samples are supposed to be stored per
    pit and location.

    """


    # build the file list
    fnames = []
    for pit_folder in sorted(glob.glob(os.path.join(data_dir, '*'))):
        # We only loop through folders
        if not os.path.isdir(pit_folder):
            continue

        pit_name = os.path.basename(pit_folder)
        for position_folder in sorted(glob.glob(os.path.join(pit_folder, '*'))):
            if not os.path.isdir(position_folder):
                continue

            fnames.append((position_folder, pit_name))

    # extract the features (parallelisation)
    samples = Parallel(n_jobs=-1, verbose=10)(
        delayed(_read_samples_per_pit_and_location_inner_loop)(position_folder, pit_name)
            for position_folder, pit_name in fnames
    )

    # Merge everythong in a row
    nb_samples1 = len(samples)
    samples = pd.concat(samples)
    nb_samples2 = samples.values.shape[0]
    assert nb_samples1 == nb_samples2
    
    return samples



class DensityHistoDistance(ClassificationExperiment):
    """We want an experiment which keeps only the density hisstogram"""

    def __init__(self, data, dist, columns):
        assert dist in ['chi2', 'euclide']
        super(DensityHistoDistance, self).__init__(data, "density_histo_distance")

        self.dist = dist
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

    
    def compare_samples(self, sample1, sample2):
        if self.dist == 'chi2':
            return self.chi2_distance(sample1, sample2)
        else:
            return self.euclidean_distance(sample1, sample2)





def test_experiment():
    # Root dir of the experiment
    DATA_DIR = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 
        '../../data/31102013_08.38/Manip_31102013_08.38')

    # Mapping to rename the groups which are supposed to be similar
    groups = {
        'A 1': 'A',
        'A 2': 'A',
        'B 1': 'B',
        'B 2': 'B'
    }

    samples = read_samples_per_pit_and_location(DATA_DIR)
    inline_merge_pits_in_conditions(samples, groups)

    classifier = DensityHistoClassifier(samples)
    classifier.run()
    classifier.visualize_samples()
    classifier.visualize_performance()



def real_experiment():
    """Launch the experiment.

    1. Read the data
    2. Extract the features
    3. Make the prediction
    4. Compute the performance
    """

    # File to treat TODO use a parameter
    DATA_DIR = '/mnt/data/BIC/06062013_14.51_1/Manip_06062013_14.51/'

    # Labels of the two cases
    COND_LIVE = 'LIVING'
    COND_FIXED = 'FIXED'

    # Assigment of a label to each pit
    groups = {
        'A 1': COND_LIVE,
        'A 2': COND_LIVE,
        'B 1': COND_LIVE,
        'B 2': COND_LIVE,
        'C 1': COND_FIXED,
        'C 2': COND_FIXED,
        'D 1': COND_FIXED,
        'D 2': COND_FIXED
    }

    # 1. a Get all the samples from the specified directory
    samples = read_samples_per_pit_and_location(DATA_DIR)
    # 1. b. Merge the common pits together
    inline_merge_pits_in_conditions(samples, groups, key='condition')


    # For verification, display the features of the two groups
    plt.figure()
    samples.groupby('condition').boxplot(rot=90)


    # Launch the classification procedure on the various extracted features and the various classifiers
    distance_scores = {}
    classifier_scores = {}
    old_distance_labels= None
    old_classifier_labels= None
    for column in ['density_hist', 'trajectory_hist', 'msd_hist', 'all']:
        for classifier_name in ('SVC', 'RF', 'SVC-PCA', 'RF-PCA', 'KNN_EUCL', 'KNN_CHI2'):
            # Compute the stuff
            logging.info("Compute for %s %s " % (column, classifier_name))
            classifier = DensityHistoClassifier(samples, classifier_name, column)
            classifier.run()

            # Store the results
            classifier_scores["%s %s" % (classifier_name, column)] = classifier._scores
            classifier_labels = classifier._true_labels
            if old_classifier_labels is not None:
                assert np.all(classifier_labels == old_classifier_labels)
            old_classifier_labels = classifier_labels

#       XXX Old code deleted, do not remember why. Need to check that later
#        for distance_name in ('chi2', 'euclide'):
#            logging.info("Compute for %s %s " % (column, distance_name))
#            dist_classifier = DensityHistoDistance(samples, distance_name, column)
#            dist_classifier.run()
#
#            dist_labels = [1]*len(dist_classifier._intra) + [-1]*len(dist_classifier._inter)
#            distance_scores["%s %s" % (distance_name, column)] = dist_classifier._intra+dist_classifier._inter
#
#            if old_distance_labels is not None:
#                assert np.all(dist_labels == old_distance_labels)
#            old_distance_labels = dist_labels
#
#
    # Display the results
    marker = {
    'SVC' :'^', 
        'RF':'<', 
        'SVC-PCA': '>', 
        'RF-PCA': "v", 
        'KNN_EUCL' : 's', 
        'KNN_CHI2': 'o'
    }
    plt.figure()
    for classifier_name in classifier_scores:
        fpr, tpr, thresholds = roc_curve(classifier_labels, classifier_scores[classifier_name])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 
                 label='%s (area = %0.2f)' % (classifier_name, roc_auc),
                 marker=marker[classifier_name.split()[0]])
#    for distance_name in distance_scores:
#        fpr, tpr, thresholds = roc_curve(dist_labels, distance_scores[distance_name])
#        roc_auc = auc(fpr, tpr)
#        plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (distance_name, roc_auc))
#        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")


# code
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    real_experiment()

    plt.show()

# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2013, 2014 LaBRI'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'romain.giot@u-bordeaux1.fr'
__status__ = 'Prototype'

