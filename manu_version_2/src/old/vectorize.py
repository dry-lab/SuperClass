#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import argparse
# from sklearn import manifold
# from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
# from sklearn.svm import SVC
# from sklearn.cross_validation import LeaveOneOut
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble.forest import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics.pairwise import additive_chi2_kernel
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction import image
# from sklearn.cluster import spectral_clustering
# from sklearn.utils.testing import SkipTest
# from sklearn.utils.fixes import sp_version
# from sklearn.cluster import MiniBatchKMeans,KMeans
# from sklearn.metrics.pairwise import pairwise_distances_argmin
# from sklearn.externals import joblib
# from sklearn.neighbors import kneighbors_graph
# from sklearn import cluster
# from sklearn import datasets
# from sklearn.semi_supervised import label_propagation

global N_CLUSTERS
global OUTPUT_DIR
global INPUT_DIR
global PREPROCESSED_MODE
global TARGET
global CLASSIFICATION_MODE
global FEATURES
global LOGFEATURES
global img_df
global obj_df
global point_df

per_image_cols = (
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
per_object_cols = (
    'ImageNumber',# numéro d’images
    'WaveTracerId', # numéro permettant de lier le Dinst au D.
    'ObjectNumber', # numéro unique
    'Tracks_Location_CenterX', # position X du premier point de la trajectoire
    'Tracks_Location_CenterY', # position y du premier point de la trajectoire
    'Diffusion_Coefficient',  # coefficient de diffusion
    'MSD_0', # ordonnée à l’origine de la MSD
    'MSE_D' # qualité du fit de la MSD
)
per_point_cols = (
    'PointNumber', # numéro unique
    'WaveTracerID', # numéro permettant de lier le Dinst au D.
    'ImageNumber', # numéro d’images
    'NumberInTrack',# position au sein de la trajectoire
    'Dinst', # valeur de Dinstantané
    'DinstL'
)

def addWellToAllDataFrame():
    global img_df, obj_df, point_df
    ImageWellMap = img_df.set_index('ImageNumber')['well'].to_dict()
    obj_df['well']=obj_df['ImageNumber'].map(ImageWellMap)
    point_df['well']=point_df['ImageNumber'].map(ImageWellMap)


def preprocess_data(df, colName):
    global TARGET
    grouped=df.groupby(TARGET)
    zscore = lambda x: (x - x.mean()) / x.std()
    transformed_data = grouped[colName].transform(zscore)  
    stdTab = pd.DataFrame(transformed_data)
    stdColName=colName+"_std"
    stdTab.columns = [stdColName]
    df=pd.concat([df, stdTab], axis=1,verify_integrity=False)
    return df


def readFeatures(features, logfeatures) :
    global img_df, obj_df, point_df
    for feature in features:
        if feature in per_image_cols:
            img_df = preprocess_data(img_df, feature)
        if feature in per_object_cols:
            obj_df = preprocess_data(obj_df, feature)
        if feature in per_point_cols:
            point_df = preprocess_data(point_df, feature)

    for feature in logfeatures:
        if feature in per_image_cols:
            img_df = logScale(img_df, feature)
        if feature in per_object_cols:
            obj_df = logScale(obj_df, feature)
        if feature in per_point_cols:
            point_df = logScale(point_df, feature)



def logScale(df, colName):
    df[colName] = np.log(df[colName])
    return df



def main():
    global N_CLUSTERS, OUTPUT_DIR, INPUT_DIR, TARGET, CLASSIFICATION_MODE, FEATURES, img_df, obj_df, point_df
    parser = argparse.ArgumentParser(description='Parameter for superclass')
    parser.add_argument('-k','--nclusters', help='Number of clusters. Default 4.', required=False)
    parser.add_argument('-c','--classification', help='Classification mode (supervised, unsupervised)', required=True)
    parser.add_argument('-o','--output', help='Output dir', required=True)
    parser.add_argument('-i','--input', help='Inpu dir where files are located', required=True)
    parser.add_argument('-p','--process', help='processing mode (normalized, other)', required=False)
    parser.add_argument('-t','--target', help='Classify pit or images', required=True)
    parser.add_argument('-b','--binning', help='freedman_std, freedman_all, freedman_max, fixed', required=False)
    parser.add_argument('-f','--features', help='array of features to use in classification', required=True)
    parser.add_argument('-l', '--log', help='Array of features to log scale', required=False)

    args = vars(parser.parse_args())
    # print args

    if args['nclusters'] is not None:
        N_CLUSTERS = args['nclusters']

    if args['classification'] is not None:
        CLASSIFICATION_MODE = args['classification']

    if args['output'] is not None:
        OUTPUT_DIR = args['output']

    if args['input'] is not None:
        INPUT_DIR = args['input']

    if args['process'] is not None:
        PREPROCESSED_MODE = args['process']

    if args['target'] is not None:
        TARGET = args['target']

    if args['binning'] is not None:
        BINNING_TYPE = args['binning']

    if args['features'] is not None:
        FEATURES = args['features'].split(',')

    if args['log'] is not None:
        LOGFEATURES = args['log'].split(',')
    else:
        LOGFEATURES=""

    per_image_file = os.path.join(INPUT_DIR, "per_image_bioinfo_Crosslink240415.csv")
    per_object_file = os.path.join(INPUT_DIR, "per_object_bioinfo_Crosslink240415.csv")
    per_point_file = os.path.join(INPUT_DIR, "per_point_bioinfo_Crosslink240415.csv")

    img_df = pd.read_csv(per_image_file, names=per_image_cols, header=None, sep=',', low_memory=False)
    obj_df = pd.read_csv(per_object_file, names=per_object_cols, header=None, sep=',', low_memory=False)
    point_df = pd.read_csv(per_point_file, names=per_point_cols, header=None, sep=',', low_memory=False)

    addWellToAllDataFrame()
    print FEATURES
    readFeatures(FEATURES, LOGFEATURES)
    print obj_df

if __name__ == "__main__":
    main()











