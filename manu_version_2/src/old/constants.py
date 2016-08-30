#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Short summary

Longer summary
"""

# imports
import numpy as np
import os


#Global vars
N_CLUSTERS = 4
OUTPUT_DIR=""
INPUT_DIR=""
CACHE_DIR =""
CLASSIFICATION_MODE = 'unsupervised'
PREPROCESSED_MODE='normalized'
IMAGEORPIT = "pits"
BINNING_TYPE='freedman_std'

#BINNING_TYPE='freedman_all'
#BINNING_TYPE='freedman_max'
#BINNING_TYPE='fixed'

NB_CONDITIONS = 0

#headers for Anne file
PER_IMAGE_COLS_ANNE = (
    'ImageNumber',
    'plate',
    'well',
    'wellNumber',
    'Image_PathNames_Path_SR_Objects',
    'Image_FileNames_Filename_SR_Objects',
    'Image_PathNames_Path_SR_Tracks',
    'Image_FileNames_Filename_SR_Tracks',
    'Points_Count',
    'Tracks_Count',
)
PER_OBJECT_COLS_ANNE = (
    'ImageNumber',# numéro d’images
    'WaveTracerId', # numéro permettant de lier le Dinst au D.
    'ObjectNumber', # numéro unique
    'Tracks_Location_CenterX', # position X du premier point de la trajectoire
    'Tracks_Location_CenterY', # position y du premier point de la trajectoire
    'Diffusion_Coefficient',  # coefficient de diffusion
    'MSD_0', # ordonnée à l’origine de la MSD
    'MSE_D' # qualité du fit de la MSD
)
PER_POINT_COLS_ANNE = (
    'PointNumber', # numéro unique
    'WaveTracerID', # numéro permettant de lier le Dinst au D.
    'ImageNumber', # numéro d’images
    'NumberInTrack',# position au sein de la trajectoire
    'Dinst', # valeur de Dinstantané
    'DinstL'
)

##type of classification associated with Anne files
per_image_cols = PER_IMAGE_COLS_ANNE
per_object_cols = PER_OBJECT_COLS_ANNE
per_point_cols = PER_POINT_COLS_ANNE


# PER_IMAGE_COLS_FIX =( 
# 'ImageNumber',
# 'plate',
# 'well',
# 'Image_PathNames_Path_SR_Objects',
# 'Image_FileNames_Filename_SR_Objects',
# 'Image_PathNames_Path_SR_Tracks',
# 'Image_FileNames_Filename_SR_Tracks',
# 'Points_Count',
# 'Tracks_Count',
# )

# PER_OBJECT_COLS_FIX = (
#     'ImageNumber',
#     'ObjectNumber',
#     'Tracks_Location_CenterX',
#     'Tracks_Location_CenterY',
#     'Diffusion_Coefficient',
#     'MSD_0',
#     'MSE_D',
#     'Alpha',
#     'Beta',
#     'MSE_Alpha',
#     'Speed',
# )

ADDITIONAL_DINST_COLS = (
'ObjectNumber',
'ImageNumber',
'NbrDinstinTracks',
'Dinst'
)

#per_image_cols = PER_IMAGE_COLS_FIX
#per_object_cols = PER_OBJECT_COLS_FIX
#per_point_cols = ADDITIONAL_DINST_COLS


# extremim of densities for the number of frames per trajectory
TRAJECTORY_FRAMES_MAX = 20
TRAJECTORY_FRAMES_MIN = 1
TRAJECTORY_FRAMES_HISTOGRAM_BINS = np.linspace(TRAJECTORY_FRAMES_MIN, TRAJECTORY_FRAMES_MAX, num=20)
TRAJECTORY_FRAMES_HISTOGRAM_LABELS = ["HIST_NB_FRAMES_%f" % _ for _ in TRAJECTORY_FRAMES_HISTOGRAM_BINS[:-1]]


groups = {
        'C1': 'LIVING',
        'C2': 'LIVING',
        'A1': 'FIXED',
        'A2': 'FIXED',
        'A3': 'FIXED',
        'A4': 'FIXED',
        'A5': 'FIXED',
        'A6': 'FIXED',
        'B1': 'FIXED',
        'B2': 'FIXED',
        'B3': 'FIXED',
        'B4': 'FIXED',
        'B5': 'FIXED',
        'B6': 'FIXED',
}


groups2 = {
        'B10': 'condition3',
        'B11': 'condition4',
        'B2': 'condition1',
        'B3': 'condition2',
        'B4': 'condition3',
        'B5': 'condition4',
        'B8': 'condition1',
        'B9': 'condition2',
        'G5': 'condition1',
        'G6': 'condition2',
        'G7': 'condition3',
        'G8': 'condition4'
}



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



# set to true if we consider that each fixed stuff correspond to the same group
#  (it is not true at all)
# CONSIDER_ALL_FIXED_AS_EQUAL = True



# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2014, LaBRI'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'romain.giot@u-bordeaux1.fr'
__status__ = 'Prototype'

