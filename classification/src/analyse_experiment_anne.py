# -*- coding: utf-8 -*-

"""Classification from Anne's dataset

The data produced by Anne are far different than the data produced by Malek
It is necessary to handle them differently ... :(


I suppose that :
 - each image corresponds to a ROI in the previous data format
 - each object corresponds to a track in the previous data format
"""

# imports
import pandas as pd
import os

from bicproject.readfile import *
from bicproject.classify import *

# Data copy pasted from the Dropbox sharing directory
# XXX Pay attention to update it if necessary
#DATA_DIR = '/mnt/data/BIC/PALM_Data'

DATA_DIR ='/home/ebouilhol/Documents/90_SuperClass/cbib-bic-labri/classification/docTest'

PER_IMAGE = os.path.join(DATA_DIR, 'per_image.csv')
PER_OBJECT = os.path.join(DATA_DIR, 'per_object.csv')
PER_IMAGE_FIX = os.path.join(DATA_DIR, 'per_image_fix.csv')
PER_OBJECT_FIX = os.path.join(DATA_DIR, 'per_object_fix.csv')
ADDITIONAL_DINST = os.path.join(DATA_DIR, 'result.csv') # Additionale file provided by Anne

ADDITIONAL_DINST_COLS = (
'ObjectNumber',
'ImageNumber',
'NbrDinstinTracks',
'Dinst'
)

PER_IMAGE_COLS_FIX =( 
'ImageNumber',
'plate',
'well',
'Image_PathNames_Path_SR_Objects',
'Image_FileNames_Filename_SR_Objects',
'Image_PathNames_Path_SR_Tracks',
'Image_FileNames_Filename_SR_Tracks',
'Points_Count',
'Tracks_Count',
)

PER_IMAGE_COLS = (
'ImageNumber',
'plate',
'well',
'Image_PathNames_Path_SR_Objects',
'Image_FileNames_Filename_SR_Objects',
'Image_PathNames_Path_SR_Tracks',
'Image_FileNames_Filename_SR_Tracks',
'Image_PathNames_Path_HR_Tracks',
'Image_FileNames_Filename_HR_Tracks',
'Points_Count',
'Tracks_Count',
)


PER_OBJECT_COLS = (
'ImageNumber',
'ObjectNumber',
'Tracks_Location_CenterX',
'Tracks_Location_CenterY',
'Diffusion_Coefficient',
'MSD_0',
'MSE_D',
'Alpha',
'Beta',
'MSE_Alpha',
'Speed',
)


# set to true if we consider that each fixed stuff correspond to the same group
#  (it is not true at all)
CONSIDER_ALL_FIXED_AS_EQUAL = True

# ===== Reading routines specific for Anne's work =====
def extract_features_of_this_ROI(df, index):
    """Extract the features for this specific ROI.
    XXX several steps are copy/pasted from readfile.py
    """
 
    # XXX WARNING it seams that min/max values previously 
    #     set are not compatible with that
    msd_features = generate_MSD_feature_vector(df['MSD_0'])
    # print "toto"
    # print msd_features
    #mse_features = generate_MSE_feature_vector(df['MSE_D'])
    diffusion_features = generate_diffusion_coefficient_features_vector(df['Diffusion_Coefficient'])
    # XXX ERROR we are unable to generate the tracking stuff
    # XXX Merge the set of features
    all_feat = pd.concat([diffusion_features, msd_features], 
                         axis=1,
                         verify_integrity=False)
    all_feat['index'] = index
    all_feat = all_feat.set_index('index')
    print all_feat
    return all_feat

def extract_features_of_each_ROI(df):
    """Extract the feature vector of each ROI (ImageNumber? or slot).
    This time we do not have any information on the number of time a particle is visible"""
    samples = []

    for ROI, data in df.groupby('ImageNumber'):
        # print "ROI : "
        # print ROI
        # print "Data : "
        # print data
        # print "APPEND : "
        # print extract_features_of_this_ROI(data, ROI)
        samples.append(extract_features_of_this_ROI(data, ROI))
        histFrdDiac, binsFrdDiac = np.histogram(data['MSD_0'], bins=MSD_HISTOGRAM_BINS)
        histFrdDiac = histFrdDiac/float(histFrdDiac.sum())
        plt.hist(histFrdDiac, binsFrdDiac)

    return pd.concat(samples)

def extract_features_of_result(df):
    """Extract the feature vector of each ROI (ImageNumber? or slot).
    This time we do not have any information on the number of time a particle is visible"""
    samples = []

    for ImageNumber, data in df.groupby('ImageNumber'):
        print "Image Number : "
        print ImageNumber
        print "Data : "
        print data['NbrDinstinTracks']
        nbdata =data['NbrDinstinTracks']
        
        hist, bins = np.histogram(nbdata, bins=bins)
        #hist = hist/float(hist.sum())
        plt.hist(hist, bins=bins, facecolor='m', alpha=0.75)
        plt.xlabel( 'Bins' )
        plt.ylabel( 'NbrDinstinTracks' )
        plt.show()



def associate_pit_to_samples(features, img_df):
    """Compute the pit of each sample and returns an array of pit"""

    # Builf the Look Up Table from an image to a pit
    pits = img_df['well']
    imgs = img_df['ImageNumber']

    LUT = dict(zip(imgs, pits))

    # Get the corresponding pit
    # (XXX as an array and not a DataFrame in order to avoid index issues)
    #

    return features.reset_index()['index'].apply(lambda idx: LUT[idx]).values




def launch_experiment(per_image_file, per_image_cols, per_object_file, additional_dist_file, groups, remove_pits=None):
    """Launch the experiment on the dataset of interest."""
 
    # Read the data files
    img_df = pd.read_csv(per_image_file, names=per_image_cols, header=None, sep=',')
    obj_df = pd.read_csv(per_object_file, names=PER_OBJECT_COLS, header=None, sep=',')
    result_df = pd.read_csv(additional_dist_file, names=ADDITIONAL_DINST_COLS, header=None, sep=',')

    # print"############################## OBJ dataframe ##################################"
    # print obj_df['MSD_0']
    # print"############################## OBJ dataframe END ##################################"

    # Get features and pits
    samples = extract_features_of_each_ROI(obj_df)
    #extract_features_of_result(result_df)
    # plotHistogram2(obj_df)

    pits = associate_pit_to_samples(samples, img_df)
    print"############################## PITS ##################################"
    print pits
    samples["pit"] = pits # XXX Add this for compatibility with previous code
                          # 
    print"############################## GROUPS ##################################"
    print groups
    print"############################## GROUPS END ##################################"


    

    # Remove the pits which are not interested
    if remove_pits:
        # search the idx of the rows to remove
        for i, pit_to_remove in enumerate(remove_pits):
            print "TO REMOVE : key =>"+i+" ; value =>"+pit_to_remove
            if 0 == i:
                idx_to_remove = pit_to_remove == samples["pit"]
            else:
                idx_to_remove = np.logical_or(idx_to_remove, pit_to_remove == samples['pit'])
        # really remove them
        samples = samples[~idx_to_remove]



     # Replace each different fixed condition by a similar one
    if CONSIDER_ALL_FIXED_AS_EQUAL:
        for key in groups:
            if groups[key].startswith('FIXED'):
                groups[key] = 'FIXED'

    

    # replace pits by condition
    inline_merge_pits_in_conditions(samples, groups, key='condition')

    print"############################## SAMPLES ##################################"
    print samples
    print"############################## SAMPLES END ##################################"

    print "CONDITION : "
    print samples['condition']

    assert not np.any(samples['condition'].isnull()), "ATTENTION, the conditions have not been set (verify the groups)"

    # XXX Here it should be exactly the same code than in the previous stuff
    samples.groupby('condition').boxplot(rot=90)
    nb_conditions = len(np.unique(samples['condition']))


    # exchange the condition label by an idx
    # If there are more than 2 conditions (living/fixed to check)
    if nb_conditions > 2:
        condition_to_num = {}
        num_to_condition = {}
        for condition in np.unique(samples['condition']):
            condition_to_num[condition] = len(condition_to_num)
            num_to_condition[ condition_to_num[condition]] = condition
        samples['condition'] = samples['condition'].apply(lambda x: condition_to_num[x])


    # Launch the classification procedure on the various extracted features and the various classifiers
    distance_scores = {}
    classifier_scores = {}
    old_distance_labels= None
    old_classifier_labels= None
    for column in ['density_hist', 'msd_hist', 'all']: #XXX trajectory_hist removed => we do not have it
        for classifier_name in ('SVC', 'RF', 'SVC-PCA', 'RF-PCA', 'KNN_EUCL', 'KNN_CHI2'):
            # Compute the stuff
            logging.info("Compute for %s %s " % (column, classifier_name))
            # print "######## LABELS Density #########"
            # print DENSITY_HISTOGRAM_LABELS
            # print "######## LABELS Density END #########"

            classifier = DensityHistoClassifier(
                samples,
                classifier_name, 
                column)
            classifier.run()

            # Store the results
            classifier_scores["%s %s" % (classifier_name, column)] = classifier._scores
            classifier_labels = classifier._true_labels
            if old_classifier_labels is not None:
                assert np.all(classifier_labels == old_classifier_labels)
            old_classifier_labels = classifier_labels


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

    if 2 == nb_conditions:
        for classifier_name in classifier_scores:
            print classifier_name 
            print classifier_scores[classifier_name]
            print "#####"
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
    else:
        for classifier_name in classifier_scores:
            print(classifier_name)
            print(classification_report(classifier_labels, classifier_scores[classifier_name]))

            cm = confusion_matrix(classifier_labels, classifier_scores[classifier_name])
            cm = cm / np.sum(cm, axis=1).astype(float)
            plt.matshow(cm, vmin=0, vmax=1)
            plt.title('Confusion matrix')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title(classifier_name)

    plt.show()
  

def experiment_fix():
    """Launch classification experiment with Anne's dataset"""

    per_image_file = PER_IMAGE_FIX
    per_object_file = PER_OBJECT_FIX
    per_image_cols = PER_IMAGE_COLS_FIX
    additional_dist_file = ADDITIONAL_DINST

    # Translate each pit to a class
    # XXX with the current dataset, each pit is different than C1, C2
    groups = {
        'C1': 'LIVING',
        'C2': 'LIVING',
        'A1': 'FIXED1',
        'A2': 'FIXED2',
        'A3': 'FIXED3',
        'A4': 'FIXED4',
        'A5': 'FIXED5',
        'A6': 'FIXED6',
        'B1': 'FIXED7',
        'B2': 'FIXED8',
        'B3': 'FIXED9',
        'B4': 'FIXED10',
        'B5': 'FIXED11',
        'B6': 'FIXED12',
    }

    launch_experiment(
        per_image_file, 
        per_image_cols, 
        per_object_file, 
        additional_dist_file,
        groups)

def experiment_orig_multiclasses():
    """Launch classification experiment with Anne's dataset"""

    per_image_file = PER_IMAGE
    per_object_file = PER_OBJECT
    per_image_cols = PER_IMAGE_COLS

    # Translate each pit to a class
    # XXX with the current dataset, each pit is different than C1, C2
    groups = {
        'B01': 'B01',
        'B02': 'B02',
        'B03': 'B03',
        'B04': 'B04-05',
        'B05': 'B04-05',
        'B06': 'B06'
    }


    pits_to_remove = ['A02', 'A03', 'A01']

    launch_experiment(per_image_file, 
                      per_image_cols, 
                      per_object_file, 
                      groups,
                      pits_to_remove)



def experiment_orig_twoclasses():
    """Launch classification experiment with Anne's dataset"""

    per_image_file = PER_IMAGE
    per_object_file = PER_OBJECT
    per_image_cols = PER_IMAGE_COLS

    # Translate each pit to a class
    # XXX with the current dataset, each pit is different than C1, C2
    groups = {
        'B01': 'B01',
        'B02': 'B02',
        'B03': 'B03',
        'B04': 'B04-05',
        'B05': 'B04-05',
        'B06': 'B06'
    }


    pits_to_remove = ['A02', 'A03', 'A01']

    launch_experiment(per_image_file, 
                      per_image_cols, 
                      per_object_file, 
                      groups,
                      pits_to_remove)


if __name__ == '__main__':
    experiment_fix()
    #experiment_orig_multiclasses()
    #experiment_orig_twoclasses()


# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2014, LaBRI'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'romain.giot@u-bordeaux1.fr'
__status__ = 'Prototype'

