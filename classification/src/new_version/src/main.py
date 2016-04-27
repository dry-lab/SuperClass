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
import pandas as pd
from Tkinter import *
#from Tkinter.filedialog import *
from tkFileDialog import *
from tkMessageBox import *
#from helpers.basics import load_config
from helpers.logger import Logger
from visualizeResults import *
from readfile import *
from classify import *
from constants import *





logger = logging.getLogger('SUPERCLASS_HELPERS')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s',"%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def group_by_condition(samples, groups):
    # Replace each different fixed condition by a similar on
    # Trouver un autre systeme
    if CONSIDER_ALL_FIXED_AS_EQUAL:	
            for key in groups:
                    if groups[key].startswith('FIXED'):	
                            groups[key] = 'FIXED'



    # replace pits by condition
    inline_merge_pits_in_conditions(samples, groups, key='condition')


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
                
def launch_experiment(per_image_file, per_image_cols,per_object_file,per_object_cols, per_point_file,per_point_cols, groups, type, remove_pits=None):
	"""Launch the experiment on the dataset of interest."""
	logger.info("Successfully launch experiment ")

	# Read the data files
	img_df = pd.read_csv(per_image_file, names=per_image_cols, header=None, sep=',', low_memory=False)
	obj_df = pd.read_csv(per_object_file, names=per_object_cols, header=None, sep=',', low_memory=False)
	point_df = pd.read_csv(per_point_file, names=per_point_cols, header=None, sep=',', low_memory=False)


	global samples
    global DENSITY_HISTOGRAM_BINS
    global DENSITY_MIN
    global DENSITY_MAX
    global DENSITY_HISTOGRAM_LABELS
    global MSD_MAX
    global MSD_MIN
    global MSD_HISTOGRAM_BINS
    global MSD_HISTOGRAM_LABELS       
    global DINST_MAX
    global DINST_MIN
    global DINST_HISTOGRAM_BINS
    global DINST_HISTOGRAM_LABELS
    if BINNING_TYPE=="freedman":

        DENSITY_MIN,DENSITY_MAX,DENSITY_HISTOGRAM_BINS,DENSITY_HISTOGRAM_LABELS,MSD_MIN,MSD_MAX,MSD_HISTOGRAM_BINS,MSD_HISTOGRAM_LABELS,DINST_MIN,DINST_MAX,DINST_HISTOGRAM_BINS,DINST_HISTOGRAM_LABELS=freedman_diaconis(obj_df,point_df)

    else:
        
        DINST_MIN=10e-5
        DINST_MAX=2.5
        DINST_HISTOGRAM_BINS=np.linspace(DINST_MIN, DINST_MAX, num=20)
        DINST_HISTOGRAM_LABELS=["HIST_DINST_%f" % _ for _ in DINST_HISTOGRAM_BINS[:-1]]
        
        
        DENSITY_MAX = 10e1
        DENSITY_MIN = 10e-5
        DENSITY_HISTOGRAM_BINS = np.logspace(-4, 1, num=20) # BINS to compute the density histogram
        DENSITY_HISTOGRAM_LABELS = ["HIST_DENSITY_%f" % _ for _ in DENSITY_HISTOGRAM_BINS[:-1]]
        
        MSD_MAX = 0.2
        MSD_MIN = -0.2
        MSD_HISTOGRAM_BINS = np.linspace(MSD_MIN, MSD_MAX, num=20)
        MSD_HISTOGRAM_LABELS = ["HIST_MSD_%f" % _ for _ in MSD_HISTOGRAM_BINS[:-1]]

    
    
    
    dinst=extract_dinst_features(point_df,DINST_MIN,DINST_MAX,DINST_HISTOGRAM_BINS,DINST_HISTOGRAM_LABELS)
    

    samples = extract_features(obj_df,DENSITY_MIN,DENSITY_MAX,DENSITY_HISTOGRAM_BINS,DENSITY_HISTOGRAM_LABELS,MSD_MIN,MSD_MAX,MSD_HISTOGRAM_BINS,MSD_HISTOGRAM_LABELS)


    samples = pd.concat([samples, dinst], axis=1,verify_integrity=False)


	pits = associate_pit_to_samples(samples, img_df)

	samples["pit"] = pits # XXX Add this for compatibility with previous code
	samples.to_csv("/Users/benjamindartigues/SuperClassTest/test_pit_merged2new2.csv",sep=",")

	# Useless
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

        if type == 'labeled':
            group_by_condition(samples,groups)
	
        btn_run_unsupervised_cell_classification_algo()
	


def btn_run_unsupervised_cell_classification_algo():
    button_unsupervised_run.configure(state=DISABLED)
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
    
    
def btn_runCellClassificationAlgo():
	button_run.configure(state=DISABLED)
        

	# Launch the classification procedure on the various extracted features and the various classifiers
	distance_scores = {}
	classifier_scores = {}
	old_distance_labels= None
	old_classifier_labels= None
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
                                MSD_HISTOGRAM_LABELS)
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
	fig=plt.figure()
        
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

def btn_choosedata():
	global var_dataDir
	
	var_dataDir = askdirectory(initialdir='/Users/benjamin/dartigues/super_class_test/')
	button_run.configure(state=NORMAL)
	button_choose.configure(state=DISABLED)
	button_load.configure(state=NORMAL)
        
        
def btn_chooseobjectdata():
	
        
	
	#var_dataDir = askdirectory(initialdir='/Users/benjamin/dartigues/super_class_test/')
        file_object_path = askopenfilename()
        
        entry_object.delete(0, END)
        entry_object.insert(0, file_object_path)
        
        if entry_object.get()!="" and entry_image.get()!="" and entry_point.get()!="":
            button_load.configure(state=NORMAL)

	
	
        
        
def btn_chooseimagedata():
	#global var_dataDir
	
	#var_dataDir = askdirectory(initialdir='/Users/benjamin/dartigues/super_class_test/')
        file_image_path = askopenfilename()
        
        entry_image.delete(0, END)
        entry_image.insert(0, file_image_path)
        

        if entry_object.get()!="" and entry_image.get()!="" and entry_point.get()!="":
            button_load.configure(state=NORMAL)
        
def btn_choosepointdata():
	#global var_dataDir
	
	#var_dataDir = askdirectory(initialdir='/Users/benjamin/dartigues/super_class_test/')
	file_point_path = askopenfilename()
        
        entry_point.delete(0, END)
        entry_point.insert(0, file_point_path)

        if entry_object.get()!="" and entry_image.get()!="" and entry_point.get()!="":
            button_load.configure(state=NORMAL)


def btn_load_data():

        button_load.configure(state=DISABLED)
	#DATA_DIR=var_dataDir
	#PER_IMAGE = os.path.join(DATA_DIR, 'per_image.csv')
	#PER_OBJECT = os.path.join(DATA_DIR, 'per_object.csv')
	#PER_IMAGE_FIX = os.path.join(DATA_DIR, 'per_image_fix.csv')
	#PER_OBJECT_FIX = os.path.join(DATA_DIR, 'per_object_fix.csv')
	#ADDITIONAL_DINST = os.path.join(DATA_DIR, 'result.csv')
	
	#PER_IMAGE = DATA_DIR + '/per_image.csv'
	#PER_OBJECT = DATA_DIR + '/per_object.csv'
	#PER_IMAGE_FIX = DATA_DIR  + '/per_image_fix.csv'
	#PER_OBJECT_FIX = DATA_DIR  + '/per_object_fix.csv'
	#ADDITIONAL_DINST = DATA_DIR + '/result.csv'
        
        #per_image_file = file_image_path.get()
	#per_object_file = file_object_path.get()
	#per_image_cols = PER_IMAGE_COLS_FIX
        #additional_dist_file=file_point_path.get()
        #global type
        #type='unlabeled'
        #binning_type='freedman'
#        
#	per_image_file="/Users/benjamindartigues/super_class_test/docTest/per_image_fix.csv"
#        per_object_file="/Users/benjamindartigues/super_class_test/docTest/per_object_fix.csv"
#        per_point_file="/Users/benjamindartigues/super_class_test/docTest/result.csv"
#        per_image_cols = PER_IMAGE_COLS_FIX
#        per_object_cols = PER_OBJECT_COLS_FIX
#        per_point_cols = ADDITIONAL_DINST_COLS
       
        
        per_image_file="/Users/benjamindartigues/super_class_test/data/per_image_bioinfo_Crosslink240415.csv"
        per_object_file="/Users/benjamindartigues/super_class_test/data/per_object_bioinfo_Crosslink240415.csv"
        per_point_file="/Users/benjamindartigues/super_class_test/data/per_point_bioinfo_Crosslink240415.csv"
        per_image_cols = PER_IMAGE_COLS_ANNE
        per_object_cols = PER_OBJECT_COLS_ANNE
        per_point_cols = PER_POINT_COLS_ANNE

	
	launch_experiment(
		per_image_file,
                per_image_cols,
		per_object_file, 
                per_object_cols,
		per_point_file,
                per_point_cols,
		groups,
                CLASS_TYPE)
        
        if (CLASS_TYPE=='unlabeled'):
            button_unsupervised_run.configure(state=NORMAL)
        else:
            button_run.configure(state=NORMAL)

if __name__ == "__main__":
    
        #global file_object_path, file_image_path,file_point_path

        
	winDataChoose = Tk()
	winDataChoose.title('Projet "SuperClass" - Cell Classification Algorithm (v0)')
	winDataChoose.focus_force()
    file_object_path= StringVar(winDataChoose)
    file_point_path = StringVar(winDataChoose)
    file_image_path= StringVar(winDataChoose)
    
    
    menubar = Menu(winDataChoose)
    winDataChoose.config(menu=menubar)
    filemenu = Menu(menubar)
    menubar.add_cascade(label="File", menu=filemenu)
    filemenu.add_command(label="Close", command=winDataChoose.quit)
        
        
        
    #label1 = Label(winDataChoose, text='Choose the directory of data, to run the classification algorithm on it: ')
	#label1.grid(row=0, column=0, pady=20, padx = 0)
	#button_choose = Button(winDataChoose, text='Choose Object files...', command = btn_choosedata)
	#button_choose.grid(row=0, column=1, pady=20, padx = 0) 
	
	
	
	samples=[]
	#var_dataDir = StringVar(winDataChoose)
	#var_dataDir.set('')
	# Additionale file provided by Anne
         
        
        
         

    labelobjet = Label(winDataChoose, text='Choose image file, to run the classification algorithm on it: ')
	labelobjet.grid(row=0, column=0, pady=20, padx = 0)
	button_object_choose = Button(winDataChoose, text='Choose Object files...', command = btn_chooseobjectdata)
	button_object_choose.grid(row=0, column=1,sticky='ew', pady=20, padx = 0)
    entry_object = Entry(winDataChoose, width=50, textvariable=file_object_path)
    entry_object.grid(row=0,column=2,padx=2,pady=2,sticky='we',columnspan=25)
    
    #Label(winDataChoose,text="File name: '").grid(row=5, column=0, sticky='e')
    #entry = Entry(winDataChoose, width=50, textvariable=file_object_path)
    #entry.grid(row=5,column=1,padx=2,pady=2,sticky='we',columnspan=25)
    #Button(winDataChoose, text="Browse", command=btn_chooseobjectdata).grid(row=1, column=27, sticky='ew', padx=8, pady=4)
        
        
        
        
    labelimage= Label(winDataChoose, text='Choose object file, to run the classification algorithm on it: ')
	labelimage.grid(row=1, column=0, pady=20, padx = 0)
    button_image_choose = Button(winDataChoose, text='Choose Images files...', command = btn_chooseimagedata)
	button_image_choose.grid(row=1, column=1, pady=20, padx = 0)
    entry_image = Entry(winDataChoose, width=50, textvariable=file_image_path)
    entry_image.grid(row=1,column=2,padx=2,pady=2,sticky='we',columnspan=25)
        
        
    labelpoint = Label(winDataChoose, text='Choose points file, to run the classification algorithm on it: ')
	labelpoint.grid(row=2, column=0, pady=20, padx = 0)
    button_point_choose = Button(winDataChoose, text='Choose points files...', command = btn_choosepointdata)
	button_point_choose.grid(row=2, column=1, pady=20, padx = 0)
    entry_point = Entry(winDataChoose, width=50, textvariable=file_point_path)
    entry_point.grid(row=2,column=2,padx=2,pady=2,sticky='we',columnspan=25)
        
        
    Label(winDataChoose, text='Load data, to run the classification algorithm on it: ').grid(row=3, column=0, pady=20, padx = 0)
	button_load = Button(winDataChoose, text='Load ...', command = btn_load_data)
	button_load.grid(row=3, column=1, pady=20, padx = 0)
	#button_load.configure(state=DISABLED)


	button_run = Button(winDataChoose, text='Run Cell Classification Algorithm (v0)', command=btn_runCellClassificationAlgo)
	button_run.configure(state=DISABLED)
	button_run.grid(row=4, column=0, columnspan=2)
        
    button_unsupervised_run = Button(winDataChoose, text='Run Unsupervised Cell Classification Algorithm (v0)', command=btn_run_unsupervised_cell_classification_algo)
	button_unsupervised_run.configure(state=DISABLED)
	button_unsupervised_run.grid(row=5, column=0, columnspan=2)

	label2 = Label(winDataChoose, text='Show result graphs: ')
	label2.grid(row=6, column=0, sticky=E)

	var_showGraphs = StringVar(winDataChoose)
	var_showGraphs.set("No") # initial value
	option_showGraphs = OptionMenu(winDataChoose, var_showGraphs, "No", "Yes")
	option_showGraphs.grid(row=6, column=1, sticky=W)



       
        
        
	canvas_results = Canvas(winDataChoose, width=550, height=400, bd=1,relief='ridge')
	canvas_results.grid(row=6, column=0, columnspan=2)
        
    btn_load_data()
        
        
        #objectcsvfile=Label(root, text="ObjectFile").grid(row=0, column=2)
        #imagecsvfile=Label(root, text="ImageFile").grid(row=1, column=2)
        #pointcsvfile=Label(root, text="PointFile").grid(row=2, column=2)
        #bar=Entry(master).grid(row=1, column=1) 

	winDataChoose.mainloop()



