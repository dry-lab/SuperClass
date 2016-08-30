#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import argparse
from collections import OrderedDict
from various_algorithm import *

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
    'Tracks_Count'
)
per_object_cols = (
    'ImageNumber',# numéro d’images
    'WaveTracerID', # numéro permettant de lier le Dinst au D.
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


groups2 = {
        'B10': '1',
        'B11': '2',
        'B2': '3',
        'B3': '4',
        'B4': '5',
        'B5': '6',
        'B8': '7',
        'B9': '8',
        'G5': '9',
        'G6': '10',
        'G7': '11',
        'G8': '12'
}


def addDinstDuration ():
    global obj_df, point_df
    point_df['WaveLength'] = point_df.groupby(['ImageNumber','WaveTracerID'])['WaveTracerID'].transform('count')
    temp = point_df.drop_duplicates(['ImageNumber', 'WaveTracerID', 'WaveLength'])
    temp2 = temp[['ImageNumber', 'WaveTracerID', 'WaveLength']]
    result = pd.merge(obj_df, temp2, how='left', on=['ImageNumber', 'WaveTracerID'])
    obj_df =  result


def addWellToAllDataFrame():
    global img_df, obj_df, point_df
    ImageWellMap = img_df.set_index('ImageNumber')['well'].to_dict()
    obj_df['well']=obj_df['ImageNumber'].map(ImageWellMap)
    point_df['well']=point_df['ImageNumber'].map(ImageWellMap)
    return ImageWellMap


def logScale(df, colName):
    df[colName+"_log"] = np.log(df[colName])
    return df

def normalize_data(df, colName):
    global TARGET
    grouped=df.groupby(TARGET)
    zscore = lambda x: (x - x.mean()) / x.std()
    transformed_data = grouped[colName].transform(zscore)  
    stdTab = pd.DataFrame(transformed_data)
    stdColName=colName+"_std"
    stdTab.columns = [stdColName]
    df=pd.concat([df, stdTab], axis=1,verify_integrity=False)
    return df


def processingOnFeatures(features, logfeatures) :
    global img_df, obj_df, point_df
    for feature in features:
        if feature in per_image_cols:
            img_df = normalize_data(img_df, feature)
        if feature in per_object_cols:
            obj_df = normalize_data(obj_df, feature)
        if feature in per_point_cols:
            point_df = normalize_data(point_df, feature)

    for feature in logfeatures:
        if feature in per_image_cols:
            img_df = logScale(img_df, feature)
        if feature in per_object_cols:
            obj_df = logScale(obj_df, feature)
        if feature in per_point_cols:
            point_df = logScale(point_df, feature)

def arrayOfInteret(features, logfeatures) :
    global img_df, obj_df, point_df
    temp = pd.DataFrame()
    for feature in features:
        if feature in per_image_cols:
            temp[feature+"_std"]=img_df[feature+"_std"]
        if feature in per_object_cols:
            temp[feature+"_std"]=obj_df[feature+"_std"]
        if feature in per_point_cols:
            temp[feature+"_std"]=point_df[feature+"_std"]

    for feature in logfeatures:
        if feature in per_image_cols:
            temp[feature+"_log"]=img_df[feature+"_log"]
        if feature in per_object_cols:
            temp[feature+"_log"]=obj_df[feature+"_log"]
        if feature in per_point_cols:
            temp[feature+"_log"]=point_df[feature+"_log"]

    temp['well']=obj_df['well']
    temp['ImageNumber']=obj_df['ImageNumber']
    return temp


def createVectors(df, target):
    vectorColumnsMap = {}
    #Create binning labels and values for all targets
    finalDf = {}
    for column in df:
        if column not in ['well', 'ImageNumber']:
            # print column
            bins = freedman_bin_width( df[column])
            # hist, bins = np.histogram(df[column], bins='fd')
            maxVal, minVal, labels = getBinInfos(bins, column)
            vectorColumnsMap[column] = {}
            vectorColumnsMap[column]['max']= maxVal
            vectorColumnsMap[column]['min']= minVal
            vectorColumnsMap[column]['labels']= labels
            vectorColumnsMap[column]['bins']= bins
    # print vectorColumnsMap # Binning values and labels for each columns


    for ROI, data in df.groupby(target):
        vector =[]
        for column in data:
            if column not in ['well', 'ImageNumber']:
                temp = generate_feature_vector(data[column],vectorColumnsMap[column]['min'],vectorColumnsMap[column]['max'],vectorColumnsMap[column]['bins'],vectorColumnsMap[column]['labels'])
                # vector = pd.concat([vector, temp], axis=1)
                vector  = np.append(vector, temp)
        finalDf[ROI]=vector
            # else faire une map ROI -> well / imageNumber pour le reafficher a la fin

    labelsList = []

    for column in vectorColumnsMap:
        if column not in ['well', 'ImageNumber']:
            labelsList  = np.append(labelsList, vectorColumnsMap[column]['labels'])
    finalDf['labels']=labelsList

    return finalDf


def generate_feature_vector(data,min,max,hist_bins,hist_labels):
    # data[data<min] = min
    # data[data>max] = max
    hist, bins = np.histogram(data,bins=hist_bins)
    hist = hist/float(hist.sum())

    # feat = OrderedDict( zip(hist_labels, hist))
    # print "FEAT"
    # print feat
    # print "HIST"
    # print hist
    # df = pd.DataFrame([hist])
    # hist = map(float, hist)
    return hist


def getBinInfos(bins, columnName):
    minValue=bins[0]
    maxValue=bins[len(bins)-1]
    labels = [columnName+"%f" % _ for _ in bins[:-1]]
    return minValue, maxValue, labels



def freedman_bin_width(data):
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")
    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")
    v25, v75 = np.percentile(data, [25, 75])
    dx = 2 * (v75 - v25) / (n ** (1 / 3))

    dmin, dmax = data.min(), data.max()
    Nbins = max(1, np.ceil((dmax - dmin) / dx))
    bins = dmin + dx * np.arange(Nbins + 1)
    return bins




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
        N_CLUSTERS = int(N_CLUSTERS)

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

    #LoadFiles
    per_image_file = os.path.join(INPUT_DIR, "per_image_bioinfo_Crosslink240415.csv")
    per_object_file = os.path.join(INPUT_DIR, "per_object_bioinfo_Crosslink240415.csv")
    per_point_file = os.path.join(INPUT_DIR, "per_point_bioinfo_Crosslink240415.csv")
    img_df = pd.read_csv(per_image_file, names=per_image_cols, header=None, sep=',', low_memory=False)
    obj_df = pd.read_csv(per_object_file, names=per_object_cols, header=None, sep=',', low_memory=False)
    point_df = pd.read_csv(per_point_file, names=per_point_cols, header=None, sep=',', low_memory=False)

    #Add the well number
    ImageWellMap = addWellToAllDataFrame()
    dfImageWellMapping = pd.DataFrame(data=ImageWellMap.values(), index=ImageWellMap.keys())
    # print "Mapping ImageNumber -> Well"
    # print dfImageWellMapping

    #Add the dinst duration column
    addDinstDuration()

    # Preprocess data, normalize, log scale.
    processingOnFeatures(FEATURES, LOGFEATURES)

    # print obj_df

    # create a df with only the columns of interest (features and logfeatures)
    dfOfInterest = arrayOfInteret(FEATURES, LOGFEATURES)

    # create the big vector of interest sort by target (currently)
    fullDict = createVectors(dfOfInterest, TARGET)

    fullDf = pd.DataFrame(data=fullDict.values(), index=fullDict.keys(), columns=fullDict['labels'])
    fullDf.to_csv(os.path.join(OUTPUT_DIR, "DataframeOfVectors.csv"), sep=",")

    fullDfWoLabels = fullDf.drop(['labels'])


    #Add well name
    # fullDfWoLabels['well']=dfImageWellMapping[0]

    # fullDfWoLabels.replace(['B10','B11','B2','B3','B4','B5','B8','B9','G5','G6','G7','G8'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], regex=True) 


    # pits = img_df['well']
    # imgs = img_df['ImageNumber']
    # LUT = dict(zip(imgs, pits))
    # pits2= fullDfWoLabels.reset_index()['index'].apply(lambda idx: LUT[idx]).values

    # fullDfWoLabels["well"] = pits2 
    print "SAMPPPLEEEEESSSSS"
    print fullDfWoLabels

    print "# Type of sample tab"
    # print type(fullDfWoLabels)

    print "# Type of well column"
    # print type(fullDfWoLabels['well'])
    # print type(fullDfWoLabels['Diffusion_Coefficient_std15.613073'])

    print "# Type of well"
    # print type(fullDfWoLabels['well'][1])
    # print type(fullDfWoLabels['Diffusion_Coefficient_std15.613073'][1])
    # fullDfWoLabels.astype(float)




    # samples_per_pit = fullDfWoLabels.groupby('well')
    # samples_per_pit.drop(['well'])
    print "samples_per_pit"
    # print samples_per_pit
    print " END samples_per_pit"
    # samples_per_pit=samples_per_pit.aggregate(np.median)
    
    # print samples_per_pit


    # fullDfWoLabels.to_csv(os.path.join(OUTPUT_DIR, "fullDfWoLabels.csv"),sep=",")
    columns = fullDfWoLabels.columns.tolist()
    # print columns
    samples = fullDfWoLabels[columns].values

    # print samples

    # np.savetxt(os.path.join(OUTPUT_DIR, "samples.csv"), samples, delimiter=",")


    index_list=fullDfWoLabels.index.tolist()
    print "List of index"
    print index_list
    result = various_algorithm_launch(samples, index_list, N_CLUSTERS, TARGET)

    print "Results"
    print result




if __name__ == "__main__":
    main()











