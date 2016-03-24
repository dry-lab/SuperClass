analyse_experiment_anne.py
	def extract_features_of_this_ROI(df, index):
	def extract_features_of_each_ROI(df):
	#def extract_features_of_result(df):
	def associate_pit_to_samples(features, img_df):
	def launch_experiment(per_image_file, per_image_cols, per_object_file, additional_dist_file, groups, remove_pits=None):
	def experiment_fix():
	def experiment_orig_multiclasses():
	def experiment_orig_twoclasses():


if __name__ == '__main__':
    experiment_fix()
    #experiment_orig_multiclasses()
    #experiment_orig_twoclasses()

Déroulement du code :

experiment_fix():
- Lecture des fichiers csv, 
- Declaration des tuples pour chaque fichier
- Definition des puits vivants ou fixés
- Appel a launch_experiment()

launch_experiment():
- data frame a partir des csv
- Appel a extract_features_of_each_ROI():
	- Pour chaque image 
	- Appel a extract_features_of_this_ROI():
		- construit un feature vector par image sur msd_0
		- le feature vector contient aussi les données de diffusion
	- Renvoi un histogramme pour chaque image dans un tableau (samples[])
- Associe les puits aux samples avec associate_pit_to_samples()
- Les differents types de fixations sont égalisés en un seul
- Lance des classifications pour tous les samples
- Plot des résultats
