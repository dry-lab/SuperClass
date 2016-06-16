#!/usr/bin/python
# -*- coding: utf-8 -*-
__user__ = 'mmluqman'
__author__ = "Muhammad Muzzamil LUQMAN"
__copyright__ = ["Copyright 2015, CBiB", "Project SuperClass"]
__credits__ = ["Muhammad Muzzamil LUQMAN", "Romain GIOT", "Emmanuel Bouilhol, Benjamin Dartigues"]
__license__ = "GPL"
__version__ = "0.0"
__maintainer__ = "Muhammad Muzzamil LUQMAN"
__email__ = 'mmluqman@u-bordeaux.fr'
__status__ = 'Prototype'

import os, sys
import argparse
import pandas as pd
from Tkinter import *
from tkFileDialog import *
from tkMessageBox import *
from visualizeResults import *
from readfile import *
from classify import *
from constants import *
from graphic_functions import *



def launch_experiment(per_image_file, per_image_cols,per_object_file,per_object_cols, per_point_file,per_point_cols, groups, remove_pits=None):
    """Launch the experiment on the dataset of interest."""
    # Read the data files
    img_df = pd.read_csv(per_image_file, names=per_image_cols, header=None, sep=',', low_memory=False)
    obj_df = pd.read_csv(per_object_file, names=per_object_cols, header=None, sep=',', low_memory=False)
    point_df = pd.read_csv(per_point_file, names=per_point_cols, header=None, sep=',', low_memory=False)

    global samples
    
    if PREPROCESSED_MODE=="normalized":##
        obj_df=preprocess_object_data(obj_df)
        point_df=preprocess_data(point_df, "Dinst")
        # point_df=preprocess_data(point_df, "DinstL")
        samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS=extract_features_bin_std(obj_df)
        dinst,DINST_HISTOGRAM_LABELS=extract_dinst_features(point_df)
        # dinst,DINST_HISTOGRAM_LABELS=extract_dinstL_features(point_df)
        # dinst,DINST_HISTOGRAM_LABELS=extract_wave_tracer_features(point_df)
        samples = pd.concat([samples, dinst], axis=1,verify_integrity=False)

    else :
        if BINNING_TYPE=="freedman_all":
            DENSITY_MIN,DENSITY_MAX,DENSITY_HISTOGRAM_BINS,DENSITY_HISTOGRAM_LABELS,MSD_MIN,MSD_MAX,MSD_HISTOGRAM_BINS,MSD_HISTOGRAM_LABELS,DINST_MIN,DINST_MAX,DINST_HISTOGRAM_BINS,DINST_HISTOGRAM_LABELS=freedman_diaconis(obj_df,point_df)
            samples = extract_features(obj_df,DENSITY_MIN,DENSITY_MAX,DENSITY_HISTOGRAM_BINS,DENSITY_HISTOGRAM_LABELS,MSD_MIN,MSD_MAX,MSD_HISTOGRAM_BINS,MSD_HISTOGRAM_LABELS)
        if BINNING_TYPE=="freedman_max":
            samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS=extract_features_bin_max(obj_df,point_df)
            #dinst,DINST_HISTOGRAM_LABELS=extract_dinst_features(point_df)
            #samples = pd.concat([samples, dinst], axis=1,verify_integrity=False)
        if BINNING_TYPE=="fixed":
            samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS=extract_features_bin_fixed(obj_df,point_df)


    pits = associate_pit_to_samples(samples, img_df)
    samples["pit"] = pits 

    if IMAGEORPIT =="pit":
        #------------------------------------------------------------ classif per pit
        samples_per_pit = samples.groupby('pit')
        samples_per_pit=samples_per_pit.aggregate(np.median)
        print samples_per_pit
        samples=samples_per_pit

    # Remove the pits which are not interested
    if remove_pits:
        samples = remove_pits(samples, remove_pits)


    if CLASSIFICATION_MODE == 'supervised':
        group_by_condition(samples,groups)
        run_cell_classification_algo(samples, DENSITY_HISTOGRAM_LABELS, MSD_HISTOGRAM_LABELS, DINST_HISTOGRAM_LABELS)
    else:  
        run_unsupervised_cell_classification_algo(samples, DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS)
        # run_unsupervised_cell_classification_algo(samples_per_pit, DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS)




def remove_pits(samples, pitsToRemove):
    for i, pit_to_remove in enumerate(remove_pits):
        print "TO REMOVE : key =>"+i+" ; value =>"+pit_to_remove
        if 0 == i:
            idx_to_remove = pit_to_remove == samples["pit"]
        else:
            idx_to_remove = np.logical_or(idx_to_remove, pit_to_remove == samples['pit'])
        # really remove them
        samples = samples[~idx_to_remove]
    return samples


def run_unsupervised_cell_classification_algo(samples, DENSITY_HISTOGRAM_LABELS, MSD_HISTOGRAM_LABELS, DINST_HISTOGRAM_LABELS):
    #button_unsupervised_run.configure(state=DISABLED)
    classifier_scores = {}
    # features=['density_hist','msd_hist','all','diff_hist']
    features=['density_hist','msd_hist']

    for column in ['all']: #features : #['diff_hist']:
        classifier = DensityHistoClassifier(
                samples,
                'K-mean', 
                column,
                CLASSIFICATION_MODE,
                DENSITY_HISTOGRAM_LABELS,
                MSD_HISTOGRAM_LABELS,
                DINST_HISTOGRAM_LABELS)
        classifier.run()

        # Store the results
        classifier_scores["%s %s" % ('K-mean', column)] = classifier._scores
    
    #Muzzamil part
	# plt1 = visualize_results_plate_class_color(classifier.result)
	# plt1.show(block=TRUE)

	#data = pd.DataFrame(np.random.random((8,12)))
	#fig = visualize_results_plate_class_membership(data)
	#fig.show()
    
    
def run_cell_classification_algo(samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS):
	#tkMessageBox.messagebox.showinfo('Cell Classification Algorithm (v0)', 'running on data: '+var_dataDir.get())
	#button_run.configure(state=DISABLED)

	# Launch the classification procedure on the various extracted features and the various classifiers
	#distance_scores = {}
	classifier_scores = {}
        classifier_scores_label={}
	#old_distance_labels= None
	old_classifier_labels= None
        #for column in ['all']:
	for column in ['density_hist', 'msd_hist', 'all']: #XXX trajectory_hist removed => we do not have it
                for classifier_name in ('SVC', 'RF', 'SVC-PCA', 'RF-PCA', 'KNN_EUCL', 'KNN_CHI2'):
			# Compute the stuff
			logging.info("Compute for %s %s " % (column, classifier_name))
			classifier = DensityHistoClassifier(
				samples,
				classifier_name, 
				column,
                CLASSIFICATION_MODE,
                DENSITY_HISTOGRAM_LABELS,
                MSD_HISTOGRAM_LABELS,
                DINST_HISTOGRAM_LABELS)
			classifier.run()

			# Store the results
			classifier_scores["%s %s" % (classifier_name, column)] = classifier._scores
                        classifier_scores_label["%s %s" % (classifier_name, column)] = classifier._scores_label
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
	fig=plt.figure()
        
	if 2 == nb_conditions:
		for classifier_name in classifier_scores:
			print classifier_name 
			print classifier_scores[classifier_name]
                        print classifier_scores_label[classifier_name]
			print "#####"
			fpr, tpr, thresholds = roc_curve(classifier_labels, classifier_scores[classifier_name])
			roc_auc = auc(fpr, tpr)
			plt.plot(fpr, tpr, 
					label='%s (area = %0.2f)' % (classifier_name, roc_auc),
					marker=marker[classifier_name.split()[0]])
                plt.plot([0, 1.5], [0, 1.05], 'k--')
		plt.xlim([0.0, 1.5])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")
                
                
	else:
		for classifier_name in classifier_scores:
			#print(classifier_name)
			#print(classification_report(classifier_labels, classifier_scores[classifier_name]))
                        print classifier_name 
			print classifier_scores[classifier_name]
                        print classifier_scores_label[classifier_name]

#			cm = confusion_matrix(classifier_labels, classifier_scores[classifier_name])
#			cm = cm / np.sum(cm, axis=1).astype(float)
#			plt.matshow(cm, vmin=0, vmax=1)
#			plt.title('Confusion matrix')
#			plt.colorbar()
#			plt.ylabel('True label')
#			plt.xlabel('Predicted label')
#			plt.title(classifier_name)

	plt.show()
        fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.pdf"),dpi=fig.dpi)


def group_by_condition(samples, groups):
    # Replace each different fixed condition by a similar on
    # Trouver un autre systeme
    # if CONSIDER_ALL_FIXED_AS_EQUAL:
    for key in groups:
        if groups[key].startswith('FIXED'):
            groups[key] = 'FIXED'

    # replace pits by condition
    key='condition'
    samples[key] = None
    for pit, cond in groups.items():
        samples[key][samples['pit'] == pit] = cond
    assert (samples[key] != -1).all(), "Some samples have no condition"

    samples.to_csv(os.path.join(OUTPUT_DIR, "test_pit_merged2new3.csv"),sep=",")

    assert not np.any(samples['condition'].isnull()), "ATTENTION, the conditions have not been set (verify the groups)"

    samples.groupby('condition').boxplot(rot=90,return_type='axes')
    global nb_conditions
    nb_conditions = len(np.unique(samples['condition']))


def main():
    global N_CLUSTERS
    global OUTPUT_DIR
    global INPUT_DIR
    global PREPROCESSED_MODE
    global IMAGEORPIT
    global CLASSIFICATION_MODE

    parser = argparse.ArgumentParser(description='Parameter for superclass')
    parser.add_argument('-k','--nclusters', help='Number of clusters. Default 4.', required=False)
    parser.add_argument('-c','--classification', help='Classification mode (supervised, unsupervised)', required=True)
    parser.add_argument('-o','--output', help='Output dir', required=True)
    parser.add_argument('-i','--input', help='Inpu dir where files are located', required=True)
    parser.add_argument('-p','--process', help='processing mode (normalized, other)', required=False)
    parser.add_argument('-j','--imageOrPit', help='Classify pit or images', required=False)
    parser.add_argument('-b','--binning', help='freedman_std, freedman_all, freedman_max, fixed', required=False)

    args = vars(parser.parse_args())
    print args

    if args['nclusters'] is not None:
        N_CLUSTERS = args['nclusters']
        print "toto"

    if args['classification'] is not None:
        CLASSIFICATION_MODE = args['classification']

    if args['output'] is not None:
        OUTPUT_DIR = args['output']

    if args['input'] is not None:
        INPUT_DIR = args['input']

    if args['process'] is not None:
        PREPROCESSED_MODE = args['process']

    if args['imageOrPit'] is not None:
        IMAGEORPIT = args['imageOrPit']

    if args['binning'] is not None:
        BINNING_TYPE = args['binning']

    per_image_file = os.path.join(INPUT_DIR, "per_image_bioinfo_Crosslink240415.csv")
    per_object_file = os.path.join(INPUT_DIR, "per_object_bioinfo_Crosslink240415.csv")
    per_point_file = os.path.join(INPUT_DIR, "per_point_bioinfo_Crosslink240415.csv")

    # per_image_file = os.path.join(INPUT_DIR, "per_image_testgraphlab.csv")
    # per_object_file = os.path.join(INPUT_DIR, "per_object_testgraphlab.csv")
    # per_point_file = os.path.join(INPUT_DIR, "per_point_testgraphlab.csv")

    launch_experiment(
        per_image_file,
                per_image_cols,
        per_object_file, 
                per_object_cols,
        per_point_file,
                per_point_cols,
        groups2)


    #    
if __name__ == "__main__":
    main()






