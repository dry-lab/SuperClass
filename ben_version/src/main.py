#!/usr/bin/python
# -*- coding: utf-8 -*-
__user__ = 'mmluqman'
__author__ = "Muhammad Muzzamil LUQMAN"
__copyright__ = ["Copyright 2015, CBiB", "Project SuperClass"]
__credits__ = ["Muhammad Muzzamil LUQMAN", "Romain GIOT"]
__license__ = "GPL"
__version__ = "0.0"
__maintainer__ = "Muhammad Muzzamil LUQMAN"
__email__ = 'mmluqman@u-bordeaux.fr'
__status__ = 'Prototype'

import os
import sys
import pandas as pd



if sys.version_info[0] == 3:
    from tkinter import *
    from tkinter import ttk
    from Tkinter.filedialog import *
    from Tkinter.messagebox import *
else:
    from Tkinter import *
    from tkFileDialog import *
    from tkMessageBox import *
    from Tkinter import *
    import ttk


#from helpers.basics import load_config
from helpers.logger import Logger
from visualizeResults import *
from readfile import *
from classify import *
from constants import *
from graphic_functions import *

#logger level 
#CRITICAL 50
#ERROR 	40
#WARNING 30
#INFO 	20
#DEBUG 	10
#NOTSET 0


logger = logging.getLogger('SUPERCLASS_HELPERS')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s',"%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


     
def launch_experiment(per_image_file, per_image_cols,per_object_file,per_object_cols, per_point_file,per_point_cols, groups, remove_pits=None):
	"""Launch the experiment on the dataset of interest."""
	logger.info("Successfully launch experiment ")

	# Read the data files
	img_df = pd.read_csv(per_image_file, names=per_image_cols, header=None, sep=',', low_memory=False)
	obj_df = pd.read_csv(per_object_file, names=per_object_cols, header=None, sep=',', low_memory=False)
	point_df = pd.read_csv(per_point_file, names=per_point_cols, header=None, sep=',', low_memory=False)

        global samples
        
        if CLASS_TYPE=='unlabeled':
            if PREPROCESSED_MODE=="standardized":##

                obj_df=preprocess_object_data(obj_df)
                point_df=preprocess_point_data(point_df)
                samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS=extract_features_bin_std(obj_df)
                dinst,DINST_HISTOGRAM_LABELS=extract_dinst_features(point_df)
                samples = pd.concat([samples, dinst], axis=1,verify_integrity=False)


            else:

                if BINNING_TYPE=="freedman_all":
                    logger.info("freedman mode detected")
                    DENSITY_MIN,DENSITY_MAX,DENSITY_HISTOGRAM_BINS,DENSITY_HISTOGRAM_LABELS,MSD_MIN,MSD_MAX,MSD_HISTOGRAM_BINS,MSD_HISTOGRAM_LABELS,DINST_MIN,DINST_MAX,DINST_HISTOGRAM_BINS,DINST_HISTOGRAM_LABELS=freedman_diaconis(obj_df,point_df)
                    samples = extract_features(obj_df,DENSITY_MIN,DENSITY_MAX,DENSITY_HISTOGRAM_BINS,DENSITY_HISTOGRAM_LABELS,MSD_MIN,MSD_MAX,MSD_HISTOGRAM_BINS,MSD_HISTOGRAM_LABELS)

                if BINNING_TYPE=="freedman_max":
                    samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS=extract_features_bin_max(obj_df,point_df)


                if BINNING_TYPE=="fixed":

                    samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS=extract_features_bin_fixed(obj_df,point_df)
        else:
            
            
            if PREPROCESSED_MODE=="standardized":##
                DINST_MIN=10e-5
                DINST_MAX=2.5
                DINST_HISTOGRAM_BINS=np.linspace(DINST_MIN, DINST_MAX, num=20)
                DINST_HISTOGRAM_LABELS=["HIST_DINST_%f" % _ for _ in DINST_HISTOGRAM_BINS[:-1]] 
                obj_df=preprocess_object_data(obj_df)
                #point_df=preprocess_point_data(point_df)
                #point_df.to_csv("/Users/benjamindartigues/SuperClassTest/pointdf_std.csv",sep=",")

                #dinst,DINST_HISTOGRAM_LABELS=extract_dinst_features(point_df)
                samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS=extract_features_bin_std(obj_df)
                #dinst,DINST_HISTOGRAM_LABELS=extract_dinst_features(point_df)
                #samples = pd.concat([samples, dinst], axis=1,verify_integrity=False)


            else:

                if BINNING_TYPE=="freedman_all":
                    logger.info("freedman mode detected")
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
        
        # XXX Add this for compatibility with previous code
        samples.to_csv("/Users/benjamindartigues/SuperClassTest/sample_file_before_classification_std.csv",sep=",")
        #dinst.to_csv("/Users/benjamindartigues/super_class_test/DINSTtest_pit_mergednew.csv",sep=",")

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
                
        if CLASS_TYPE == 'labeled':
            group_by_condition(samples,groups)            
            run_cell_classification_algo(samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS)
        else:  
            run_unsupervised_cell_classification_algo(samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS)
	


def run_unsupervised_cell_classification_algo(samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS):
    #button_unsupervised_run.configure(state=DISABLED)
    classifier_scores = {}
    features=['density_hist','msd_hist','all','diff_inst']
    for column in ['all']:
        classifier = DensityHistoClassifier(
                samples,
                'K-mean', 
                column,
                CLASS_TYPE,
                DENSITY_HISTOGRAM_LABELS,
                MSD_HISTOGRAM_LABELS,
                DINST_HISTOGRAM_LABELS)
        classifier.run()

        # Store the results
        classifier_scores["%s %s" % ('K-mean', column)] = classifier._scores
    
        #Muzzamil part
	plt1 = visualize_results_plate_class_color(classifier.result)
	plt1.show(block=FALSE)

	#data = pd.DataFrame(np.random.random((8,12)))
	#fig = visualize_results_plate_class_membership(data)
	#fig.show()
    
    
def run_cell_classification_algo(samples,DENSITY_HISTOGRAM_LABELS,MSD_HISTOGRAM_LABELS,DINST_HISTOGRAM_LABELS):
	#tkMessageBox.messagebox.showinfo('Cell Classification Algorithm (v0)', 'running on data: '+var_dataDir.get())
	#button_run.configure(state=DISABLED)
        

	# Launch the classification procedure on the various extracted features and the various classifiers

	classifier_scores = {}
        classifier_scores_label={}
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
                                CLASS_TYPE,
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
        fig.savefig('/Users/benjamindartigues/super_class_test/src/roc_curve.pdf',dpi=fig.dpi)

def group_by_condition(samples, groups):
    # Replace each different fixed condition by a similar on
    # Trouver un autre systeme
    if CONSIDER_ALL_FIXED_AS_EQUAL:	
            for key in groups:
                    if groups[key].startswith('FIXED'):	
                            groups[key] = 'FIXED'



    # replace pits by condition
    inline_merge_pits_in_conditions(samples, groups, key='condition')

    samples.to_csv("/Users/benjamindartigues/super_class_test/test_pit_merged2new3.csv",sep=",")

    assert not np.any(samples['condition'].isnull()), "ATTENTION, the conditions have not been set (verify the groups)"
    # XXX Here it should be exactly the same code than in the previous stuff

    ######Here we produced 
    samples.groupby('condition').boxplot(rot=90,return_type='axes')
    global nb_conditions
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
           

if __name__ == "__main__":
    samples=[]
    launch_experiment(
		per_image_file,
                per_image_cols,
		per_object_file, 
                per_object_cols,
		per_point_file,
                per_point_cols,
		groups)





