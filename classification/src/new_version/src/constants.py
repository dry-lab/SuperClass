#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Short summary

Longer summary
"""

# imports
import numpy as np


# Columns to read in the CSV file
ROI_COL = 'ROI'
TRACE_COL = 'Trace'
DENSITY_COL = 'D(µm²/s)'
MSD_COL = 'MSD(0)'
MSE_COL = 'MSE'



#type of classification
CLASS_TYPE='unlabeled'
#CLASS_TYPE='labeled'


#type of binning method
#BINNING_TYPE='freedman'
BINNING_TYPE='none'


# extremim of densities for the number of frames per trajectory
TRAJECTORY_FRAMES_MAX = 20
TRAJECTORY_FRAMES_MIN = 1
TRAJECTORY_FRAMES_HISTOGRAM_BINS = np.linspace(TRAJECTORY_FRAMES_MIN, TRAJECTORY_FRAMES_MAX, num=20)
TRAJECTORY_FRAMES_HISTOGRAM_LABELS = ["HIST_NB_FRAMES_%f" % _ for _ in TRAJECTORY_FRAMES_HISTOGRAM_BINS[:-1]]


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




PER_OBJECT_COLS_FIX = (
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



ADDITIONAL_DINST_COLS = (
'ObjectNumber',
'ImageNumber',
'NbrDinstinTracks',
'Dinst'
)


## extremum of densities for the histograms
#DENSITY_MAX = 10e1
#DENSITY_MIN = 10e-5
#DENSITY_HISTOGRAM_BINS = np.logspace(-4, 1, num=20) # BINS to compute the density histogram
#DENSITY_HISTOGRAM_LABELS = ["HIST_DENSITY_%f" % _ for _ in DENSITY_HISTOGRAM_BINS[:-1]]
#
#
## extremum of msd for the histograms
#MSD_MAX = 0.2
#MSD_MIN = -0.2
#MSD_HISTOGRAM_BINS = np.linspace(MSD_MIN, MSD_MAX, num=20)
##print ("MSD_HISTOGRAM_BINS", MSD_HISTOGRAM_BINS)
#MSD_HISTOGRAM_LABELS = ["HIST_MSD_%f" % _ for _ in MSD_HISTOGRAM_BINS[:-1]]

# MSD_MAX = 0.5
# MSD_MIN = -0.5
# MSD_HISTOGRAM_BINS = np.linspace(MSD_MIN, MSD_MAX, num=20)
# MSD_HISTOGRAM_LABELS = ["HIST_MSD_%f" % _ for _ in MSD_HISTOGRAM_BINS[:-1]]



# set to true if we consider that each fixed stuff correspond to the same group
#  (it is not true at all)
CONSIDER_ALL_FIXED_AS_EQUAL = True



# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2014, LaBRI'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'romain.giot@u-bordeaux1.fr'
__status__ = 'Prototype'

