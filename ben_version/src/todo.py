#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
##----------------##
## Import library ##
##----------------##

#------------------------------------------------- /!\ check for useless library
import warnings

import logging
from logging.handlers import RotatingFileHandler

# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import additive_chi2_kernel
from sklearn.decomposition import PCA


# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger()
# on met le niveau du logger à DEBUG, comme ça il écrit tout
logger.setLevel(logging.DEBUG)

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
# création d'un handler qui va rediriger une écriture du log vers
# un fichier en mode 'append', avec 1 backup et une taille max de 1Mo
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
# on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
# créé précédement et on ajoute ce handler au logger
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# création d'un second handler qui va rediriger chaque écriture de log
# sur la console
steam_handler = logging.StreamHandler()
steam_handler.setLevel(logging.DEBUG)
logger.addHandler(steam_handler)
 
# Après 3 heures, on peut enfin logguer
# Il est temps de spammer votre code avec des logs partout :
# logger.info('Hello')
# logger.warning('Testing %s', 'foo')



logger.info('Reading files...')
DATA_DIR ='/home/ebouilhol/Documents/90_SuperClass/cbib-bic-labri/classification/docTest'
per_image_file = os.path.join(DATA_DIR, 'per_image_30.csv')
per_object_file = os.path.join(DATA_DIR, 'per_object_30.csv')
result_file = os.path.join(DATA_DIR, 'result_30.csv')


PER_IMAGE_COLS = (
'ImageNumber',
'plate',
'pit',
'Image_PathNames_Path_SR_Objects',
'Image_FileNames_Filename_SR_Objects',
'Image_PathNames_Path_SR_Tracks',
'Image_FileNames_Filename_SR_Tracks',
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

RESULT_COL = (
'ObjectNumber',
'ImageNumber',
'NbrDinstinTracks',
'Dinst'
)

pits_state = {
        # 'C1': 'LIVING',
        # 'C2': 'LIVING',
        'A1': 'LIVING',
        'A2': 'FIXED2',
        # 'A3': 'FIXED3',
        # 'A4': 'FIXED4',
        # 'A5': 'FIXED5',
        # 'A6': 'FIXED6',
        # 'B1': 'FIXED7',
        # 'B2': 'FIXED8',
        # 'B3': 'FIXED9',
        # 'B4': 'FIXED10',
        # 'B5': 'FIXED11',
        # 'B6': 'FIXED12',
    }


##----------------##
## read csv files ##
##----------------##
# Garder que ce qui est utile dans le code de lecture des fichiers, l'importer ici

logger.info('Reading per image...')
img_df = pd.read_csv(per_image_file, names=PER_IMAGE_COLS, header=None, sep=',')

logger.info('Reading per object')
obj_df = pd.read_csv(per_object_file, names=PER_OBJECT_COLS, header=None, sep=',')

logger.info('Reading result')
result_df = pd.read_csv(result_file, names=RESULT_COL, header=None, sep=',')

logger.info('Merging dataframes')
global_df = pd.merge(obj_df, img_df, on = 'ImageNumber');
# global_df = pd.merge(global_df, result_df, on = 'ImageNumber')

# print (global_df)
global_df.to_csv('global_df.csv')


##----------------##
## merging datas  ##
##----------------##
# Faire une classe de collection et merge des données,
# - Collection par image
# - Collection par trajectoire
# Faire une classe de création d'histogrammes
# - Creer les bins séparement des histogrammes avec différentes formules possibles donc freedman diaconis
# - Tester la méthode : garder le plus grand nombre de bins par image sur la totalité des images, ce qui implique : 
# 		* Comparer le nombre de bins necessaire a chaque image
# 		* Garder le plus grand
# 		* Creer ensuite les tableaux de bins pour chaque image
# 		* Remplir les cases vides par des 0
# 		=> il y a surement des étapes factorisables

##----------------##
## classification ##
##----------------##
# - Définir les données necessaire en entrée des classifieurs
# - Faire tourner les classifieurs et les methodes de validation
# - Prévoir des sorties graphiques et non graphiques
# - Prévoir des sorties compatibles avec l'interface
# 
