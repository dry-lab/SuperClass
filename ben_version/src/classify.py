#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Try the classification stuff

"""
import exceptions
import logging
logging.basicConfig(level=logging.DEBUG)

# imports
import time
import pandas as pd
pd.set_option('display.max_rows', 120)
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy as sp

from time import time
from collections import OrderedDict
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import additive_chi2_kernel
from sklearn.decomposition import PCA
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.externals import joblib
from constants import *




def inline_merge_pits_in_conditions(samples, groups, key='position'):
    """Merge several pits together in order to be able to says that such group of pits
    correspond to the same kind of experiment
    """

    samples[key] = None

    for pit, cond in groups.items():

        samples[key][samples['pit'] == pit] = cond

    assert (samples[key] != -1).all(), "Some samples have no condition"





class ClassificationExperiment(object):
    """Manage the classification experiment"""

    def __init__(self, data_to_copy, label):
        """Initialize the various objects.
        
        data_to_copy:
            original data object we want to copy in order to modify it
        label:
            Label of the experiment
        """


        self._data = pd.DataFrame(data_to_copy, copy=True)
        self._label = label



    
    def get_columns_to_remove(self):
        """Get the list of columns to remove from the dataset"""
        return []

    def _remove_unused_columns(self):
        """Remove the columns not intersing expect if they are pit, position or condition"""
        for column in self.get_columns_to_remove():
            if column not in ['pit', 'position', 'condition']:
                del self._data[column]


    def _compute_comparison_scores(self, key='condition', ):
        """Try to compare the samples.
        XXX Need to improve that when we will have more data.
        XXX Allow to change the comparison method
        XXX we do not assume that the distance measure is symetric, so 
        the comparison of r1 to r2 and r2 to r2 can give different scores
        """
        samples = self._data

        grouped = samples.groupby(key)
        intra_scores = []
        inter_scores = []

        # We do not want columns other than the ones of extracted features
        columns = samples.columns.tolist()
        for column in ['pit', 'position', 'condition']:
            del columns[columns.index(column)]

        for name, group in grouped:
            logging.info('Compute distance within group ' + name)


            # Compute intra scores
            for idx1, row1 in group.iterrows():
                for idx2, row2 in group.iterrows():
                    # do not compare identic rows
                    if idx1 == idx2:
                        continue
                    else:
                        logging.debug('\tCompare %s to %s' % (idx1, idx2))
                        intra_scores.append(self.compare_samples(row1[columns], row2[columns]))

        for name1, group1 in grouped:
            for name2, group2 in grouped:
                if name1 == name2:
                    continue
                else:
                    logging.info("Compute distance between " + name + " and " + name2)
                    for idx1, row1 in group1.iterrows():
                        for idx2, row2 in group2.iterrows():
                            logging.debug('\tCompare %s to %s' % (idx1, idx2))
                            inter_scores.append(self.compare_samples(row1[columns], row2[columns]))
        

        self._intra = intra_scores
        self._inter = inter_scores

    # Unused ?
    # def chi2_distance(self, sample1, sample2):
    #     """Compare two samples together using the Chi2 method"""

    #     return ((sample1 - sample2)*(sample1 - sample2) / (sample1+sample2 + 0.0001)).sum()

    # def euclidean_distance(self, sample1, sample2):
    #     """Compare two samples using the euclidean distance"""
    #     return ((sample1-sample2)*(sample1-sample2)**2).sum()

    # def visualize_samples(self, key='condition'):
    #     """Visualize the samples depending on the experimental conditions"""
    #     samples = self._data
    #     samples.groupby(key).boxplot(rot=90, grid=True)
    #     plt.savefig('data_distribution.pdf')


    def visualize_performance(self):
        intra = self._intra
        inter = self._inter

        labels = [1]*len(intra) + [-1]*len(inter)
        scores = intra+inter

        self._common_visualize_performance( labels, scores)

        plt.figure()
        plt.boxplot([intra, inter])
        plt.xticks([1, 2], ['intra', 'inter'])
        plt.title('Distribution of scores')
        plt.savefig('comparison_score_distribution.pdf')


        plt.figure()
        start = np.min(np.min(intra), np.min(inter))
        end = np.max(np.max(intra), np.max(inter))
        intra_hist, intra_bin = np.histogram(intra,50, (start, end))
        inter_hist, inter_bin = np.histogram(inter,50, (start, end))


        plt.plot(intra_bin[:-1], intra_hist/float(intra_hist.sum()), label='intra', color='blue')
        plt.plot(inter_bin[:-1], inter_hist/float(inter_hist.sum()), label='inter', color='red')
        plt.legend()
        plt.xlabel('Comparison scores')
        plt.ylabel('Probability')
        plt.title('Score distribution')



    def _common_visualize_performance(self, labels, scores):
        plt.figure()
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive (A) Rate')
        plt.ylabel('True Positive (A) Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        

    def run(self):
        self._remove_unused_columns()
        self._compute_comparison_scores()


class ClassificationExperimentMachineLearning(ClassificationExperiment):
    """Instead of using distance functions, use machine learning (ie, need to compute a model)"""

    def __init__(self, data, label, classifier):
        assert classifier in ('SVC', 'RF', 'SVC-PCA', 'RF-PCA', 'KNN_EUCL', 'KNN_CHI2','K-mean')
        super(ClassificationExperimentMachineLearning, self).__init__(data, label)
        self._selected_classifier = classifier

    def run(self):
        logging.info('Classifier running with %s ' , self._label)
        #super(ClassificationExperimentMachineLearning, self).run()
        self._remove_unused_columns()
        if self.type=='labeled':
            self._compute_comparison_scoress_with_cross_validation()
        else:
            self._compute_comparison_scores_unlabeled()
            
    def _various_algorithm(self,samples,pits):  
        import time

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import cluster, datasets
        from sklearn.neighbors import kneighbors_graph
        from sklearn.preprocessing import StandardScaler
        np.random.seed(0)

        # Generate datasets. We choose the size big enough to see the scalability
        # of the algorithms, but not too big to avoid too long running times
#        n_samples = 1500
#        noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
#                                              noise=.05)
#        print noisy_circles
#        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
#        blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
#        no_structure = np.random.rand(n_samples, 2), None

        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)

        clustering_names = [
            'Kmeans','MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
            'SpectralClustering', 'Ward', 'AgglomerativeClustering',
            'DBSCAN', 'Birch']

        plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)

        plot_num = 1

        #datasets = [noisy_circles, noisy_moons, blobs, no_structure,samples]
        datasets = [samples]

        for i_dataset, dataset in enumerate(datasets):
            
            if i_dataset==len(datasets)-1:
                print i_dataset
                X=samples
                y=pits
            else:
                X, y = dataset
            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # create clustering estimators
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(n_clusters=4)
           
            k_means = cluster.KMeans(init='k-means++', n_clusters=4, n_init=100)
 

        
            ward = cluster.AgglomerativeClustering(n_clusters=4, linkage='ward',
                                                   connectivity=connectivity)
            spectral = cluster.SpectralClustering(n_clusters=4,
                                                  eigen_solver='arpack',
                                                  affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=.2)
            affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                               preference=-200)

            average_linkage = cluster.AgglomerativeClustering(
                linkage="average", affinity="cityblock", n_clusters=4,
                connectivity=connectivity)
            

            birch = cluster.Birch(n_clusters=4)
            clustering_algorithms = [
                k_means,two_means, affinity_propagation, ms, spectral, ward, average_linkage,
                dbscan, birch]

            for name, algorithm in zip(clustering_names, clustering_algorithms):
                # predict cluster memberships
                t0 = time.time()
                algorithm.fit(X)
                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)
                if i_dataset==len(datasets)-1:
                    print name
                    #print y_pred  
                    classif_df=pd.DataFrame(y_pred,index=np.arange(1,len(y_pred)+1))
                    classif_df.columns = ['scores']

                    pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
                    pit_df.columns = ['pits']

                    result_per_pit = pd.concat([classif_df, pit_df], axis=1,verify_integrity=False)
                    #print "result per pit in kmean function"
                    print result_per_pit

                # plot
                plt.subplot(1, len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)
                plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

                if hasattr(algorithm, 'cluster_centers_'):
                    centers = algorithm.cluster_centers_
                    center_colors = colors[:len(centers)]
                    plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                plot_num += 1

        plt.show()
        
    
    
    
    def _dbscan(self,samples,pits):
        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.preprocessing import StandardScaler


#        ##############################################################################
#        # Generate sample data
#        centers = [[1, 1], [-1, -1], [1, -1]]
#        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                                    random_state=0)
#
#        X = StandardScaler().fit_transform(X)

        ##############################################################################
        # Compute DBSCAN
        db = DBSCAN(eps=0.3, min_samples=10).fit(samples)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(pits, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(pits, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(pits, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(pits, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(pits, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(samples, labels))

        ##############################################################################
        # Plot result
        

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    
    def _agglomerative_clustering(self,samples,pits,type):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.feature_extraction.image import grid_to_graph

        if type=='structured':
            X = np.reshape(samples, (-1, 1))
            print X
            connectivity = grid_to_graph(*samples.shape)
            print connectivity
            # Compute clustering
            print("Compute structured hierarchical clustering...")
            st = time()
            n_clusters = 4  # number of regions
            ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                           connectivity=connectivity)
            clf =ward.fit_predict(X)
            print clf.reshape((57,119))
            
            clf_df=pd.DataFrame(clf,index=np.arange(1,len(clf)+1))
            clf_df.columns = ['scores']

            pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
            pit_df.columns = ['pits']
            pit = pit_df['pits'].values

            result_per_pit = pd.concat([clf_df, pit_df], axis=1,verify_integrity=False)
            print "result per pit in kmean function"
            print result_per_pit
            self._scores = clf
            label = np.reshape(ward.labels_, samples.shape)
            print("Elapsed time: ", time() - st)
            print("Number of pixels: ", label.size)
            print("Number of clusters: ", np.unique(label).size)

            ###############################################################################
            # Plot the results on an image
            plt.figure(figsize=(5, 5))
            plt.imshow(samples, cmap=plt.cm.gray)
            for l in range(n_clusters):
                plt.contour(label == l, contours=1,
                            colors=[plt.cm.spectral(l / float(n_clusters)), ])
            plt.xticks(())
            plt.yticks(())
            plt.show()
        else:
            
            #samples = np.concatenate([samples, samples])
            #pits = np.concatenate([pits, pits], axis=0)

            # 2D embedding of the digits dataset
            print("Computing embedding")
            X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(samples)
            print("Done.")


            
            for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):

                if metric=="cosine":
                    for linkage in ('average', 'complete'):

                        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=4,affinity=metric)
                        #t0 = time()
                        clf=clustering.fit_predict(X_red)
                        print clf
                        clf_df=pd.DataFrame(clf,index=np.arange(1,len(clf)+1))
                        clf_df.columns = ['scores']

                        pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
                        pit_df.columns = ['pits']
                        pit = pit_df['pits'].values

                        result_per_pit = pd.concat([clf_df, pit_df], axis=1,verify_integrity=False)
                        print "result per pit in kmean function"
                        print result_per_pit

                        #print("%s : %.2fs" % (linkage, time() - t0))
                        x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
                        X_red = (X_red - x_min) / (x_max - x_min)

                        plt.figure(figsize=(6, 4))
                        for i in range(X_red.shape[0]):
                            print clustering.labels_[i]
                            plt.text(X_red[i, 0], X_red[i, 1], str(pit[i]),
                                     color=plt.cm.spectral(clustering.labels_[i] / 10.),
                                     fontdict={'weight': 'bold', 'size': 9})

                        plt.xticks([])
                        plt.yticks([])
                        title = "'{0}' metric with '{1}'".format(metric, linkage)
                        #if title is not None:
                        plt.title(title, size=17)
                        #plt.title("%s metric with %s linkage" % metric,linkage, size=17)
                        plt.axis('off')
                        plt.tight_layout()

                elif metric=="euclidean":
                    for linkage in ('ward', 'average', 'complete'):

                        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=4,affinity=metric)
                        #t0 = time()
                        clf=clustering.fit_predict(X_red)
                        print clf
                        clf_df=pd.DataFrame(clf,index=np.arange(1,len(clf)+1))
                        clf_df.columns = ['scores']

                        pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
                        pit_df.columns = ['pits']
                        pit = pit_df['pits'].values

                        result_per_pit = pd.concat([clf_df, pit_df], axis=1,verify_integrity=False)
                        print "result per pit in kmean function"
                        print result_per_pit

                        #print("%s : %.2fs" % (linkage, time() - t0))
                        x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
                        X_red = (X_red - x_min) / (x_max - x_min)

                        plt.figure(figsize=(6, 4))
                        for i in range(X_red.shape[0]):
                            print clustering.labels_[i]
                            plt.text(X_red[i, 0], X_red[i, 1], str(pit[i]),
                                     color=plt.cm.spectral(clustering.labels_[i] / 10.),
                                     fontdict={'weight': 'bold', 'size': 9})

                        plt.xticks([])
                        plt.yticks([])
                        title = "'{0}' metric with '{1}'".format(metric, linkage)
                        #if title is not None:
                        plt.title(title, size=17)

                        #plt.title("%s metric with %s linkage" % metric,linkage, size=17)
                        plt.axis('off')
                        plt.tight_layout()

                else:
                    for linkage in ('average', 'complete'):

                        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=4,affinity=metric)
                        #t0 = time()
                        clf=clustering.fit_predict(X_red)
                        print clf
                        clf_df=pd.DataFrame(clf,index=np.arange(1,len(clf)+1))
                        clf_df.columns = ['scores']

                        pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
                        pit_df.columns = ['pits']
                        pit = pit_df['pits'].values

                        result_per_pit = pd.concat([clf_df, pit_df], axis=1,verify_integrity=False)
                        print "result per pit in kmean function"
                        print result_per_pit

                        #print("%s : %.2fs" % (linkage, time() - t0))
                        x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
                        X_red = (X_red - x_min) / (x_max - x_min)

                        plt.figure(figsize=(6, 4))
                        for i in range(X_red.shape[0]):
                            print clustering.labels_[i]
                            plt.text(X_red[i, 0], X_red[i, 1], str(pit[i]),
                                     color=plt.cm.spectral(clustering.labels_[i] / 10.),
                                     fontdict={'weight': 'bold', 'size': 9})

                        plt.xticks([])
                        plt.yticks([])
                        title = "'{0}' metric with '{1}'".format(metric, linkage)
                        #if title is not None:
                        plt.title(title, size=17)
                        #plt.title("%s metric with %s linkage" % metric,linkage, size=17)
                        plt.axis('off')
                        plt.tight_layout()

            plt.show()
        
        
    def _k_mean(self,samples,pits):

        print samples
        n_clusters =4
        k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100)
        k_means.fit(samples)
        k_means_labels = k_means.labels_

        print k_means.labels_
        k_means_cluster_centers = k_means.cluster_centers_
        classif=k_means.predict(samples)
        
        ####SAVE THE MODEL 
        
        #joblib.dump(k_means, 'kmean.pkl')
        
        ###RELOAD THE MODEL 
        
        #k_means = joblib.load('kmean.pkl') 
        
        batch_size = 100
        
        self._scores = classif
        
        
        classif_df=pd.DataFrame(classif,index=np.arange(1,len(classif)+1))
        classif_df.columns = ['scores']

        pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
        pit_df.columns = ['pits']

        result_per_pit = pd.concat([classif_df, pit_df], axis=1,verify_integrity=False)
        #print "result per pit in kmean function"
        #print result_per_pit
        #result_per_pit=pd.DataFrame(result_per_pit,index=np.arange(0,len(result_per_pit)))
        self.result = pd.concat([result_per_pit.groupby('pits')['scores'].sum(), result_per_pit.groupby('pits')['scores'].count()], axis=1,verify_integrity=False)
        self.result.columns = ['scores','count']
        self.result.to_csv("/Users/benjamindartigues/SuperClassTest/final_result.csv",sep=",")
    #result_per_pit.columns = ['pits','scores','count']
        result_per_pit.to_csv("/Users/benjamindartigues/SuperClassTest/pre_result_simple_k_means_freedman.csv",sep=",")
        #print result_per_pit.reindex(range(119))
        #print result_per_pit[:1]
        
        result_per_pit= result_per_pit[['scores','pits']].values
        result_per_pit_df=pd.DataFrame(result_per_pit,index=np.arange(len(result_per_pit)))
        result_per_pit_df.columns =['scores','pits']
        #result_per_pit_df = result_per_pit_df.set_index('index')
        
        grouped = result_per_pit_df.groupby('pits')
        
        for pit,cluster in grouped:
            print cluster
        print result_per_pit_df
        
        ############################################################################################
                #try to clusterize results
        ############################################################################################
#        
#        DENSITY_RESULT_BINS=[0,1,2,3,4,5]
#        #print DENSITY_RESULT_BINS
#        DENSITY_RESULT_LABELS = ["HIST_RESULT_%d" % _ for _ in DENSITY_RESULT_BINS]
#        #print DENSITY_RESULT_LABELS
#        sample_result=[]
#        counter=1
#        for ROI, data in result_per_pit_df.groupby('pits'):
#            print data['scores']
#            print ROI
##            #print ROI
#            hist, bins = np.histogram(data['scores'], bins=DENSITY_RESULT_BINS)
##            
##            #hist = hist/float(hist.sum())
##            #print hist
##        # Build the features
#            feat = OrderedDict( zip(DENSITY_RESULT_LABELS, hist))
#            print feat
##
#            df = pd.DataFrame([feat])
#            print df
##
#            df = df.reindex_axis(feat.keys(), axis=1) #order seems eroneous
#            print df
##
#            df['index'] = counter
##
#            df = df.set_index('index')
#            print df
#            sample_result.append(df)
#            counter+=1
#        sample= pd.concat(sample_result)
#        #sample=pd.DataFrame(sample_result)
#        sample.to_csv("/Users/benjamindartigues/SuperClassTest/pre_result.csv",sep=",")
#        sample= sample[sample.columns.tolist()].values
#        print sample
#        k_means.fit(sample)
#        classif=k_means.predict(sample)
#        #classif=k_means.fit_predict(sample)
#        print classif
        ############################################################################################
                #try to clusterize results end
        ############################################################################################
        
        
        ############################################################################################
        # Compute clustering with MiniBatchKMeans
        ############################################################################################
        
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                              n_init=100, max_no_improvement=10, verbose=0)
        
        
        
        t0 = time()

        mbk.fit(samples)
        t_mini_batch = time() - t0
        mbk_means_labels = mbk.labels_
        mbk_means_cluster_centers = mbk.cluster_centers_
        mbk_means_labels_unique = np.unique(mbk_means_labels)
        
        
        print mbk.labels_
        classif=mbk.predict(samples)
        print classif
        
        classif_df=pd.DataFrame(classif,index=np.arange(1,len(classif)+1))
        classif_df.columns = ['scores']

        pit_df=pd.DataFrame(pits,index=np.arange(1,len(pits)+1))
        pit_df.columns = ['pits']

        result_per_pit = pd.concat([classif_df, pit_df], axis=1,verify_integrity=False)
        print "result per pit in Mini Batch KMeans function"
        print result_per_pit
        result_per_pit.to_csv("/Users/benjamindartigues/SuperClassTest/pre_result_batch_k_means_freedman.csv",sep=",")

        ##############################################################################
        # Plot result

        fig = plt.figure(figsize=(8, 3))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        colors = ['#4EACC5', '#FF9C34', '#4E9A06','#4E9A06']#,'#555555']

        # KMeans
        ax = fig.add_subplot(1, 3, 1)
        for k, col in zip(range(n_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            ax.plot(samples[my_members, 0], samples[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)
        ax.set_title('KMeans')
        ax.set_xticks(())
        ax.set_yticks(())
        
        
        # We want to have the same colors for the same cluster from the
        # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
        # closest one.

        order = pairwise_distances_argmin(k_means_cluster_centers,
                                          mbk_means_cluster_centers)
        
        
        # MiniBatchKMeans
        ax = fig.add_subplot(1, 3, 2)
        for k, col in zip(range(n_clusters), colors):
            my_members = mbk_means_labels == order[k]
            cluster_center = mbk_means_cluster_centers[order[k]]
            ax.plot(samples[my_members, 0], samples[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)
        ax.set_title('MiniBatchKMeans')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
                 (t_mini_batch, mbk.inertia_))
        
    
        #plt.show()
    def _spectral_clustering(self,samples):
        if sp_version < (0, 12):
            raise SkipTest("Skipping because SciPy version earlier than 0.12.0 and "
                   "thus does not include the scipy.misc.face() image.")


        # load the raccoon face as a numpy array
#        try:
#            face = sp.face(gray=True)
#            print "first face"
#            print face
#        except AttributeError:
#            # Newer versions of scipy have face in misc
#            from scipy import misc
#            face = misc.face(gray=True)
#            print "first face"
#            print face

        # Resize it to 10% of the original size to speed up the processing
#        face = sp.misc.imresize(face, 0.10) / 255.
#        print "second face"
#        print face
        # Convert the image into a graph with the value of the gradient on the
        # edges.
        graph = image.img_to_graph(samples)


        # Take a decreasing function of the gradient: an exponential
        # The smaller beta is, the more independent the segmentation is of the
        # actual image. For beta=1, the segmentation is close to a voronoi
        beta = 5
        eps = 1e-6
        graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

        # Apply spectral clustering (this step goes much faster if you have pyamg
        # installed)
        N_REGIONS = 4

        #############################################################################
        # Visualize the resulting regions

        for assign_labels in ('kmeans', 'discretize'):
            t0 = time.time()
            labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                         assign_labels=assign_labels, random_state=1)
            sample=pd.DataFrame(labels)
            sample.to_csv("/Users/benjamindartigues/SuperClassTest/spectral_result.csv",sep=",")
            t1 = time.time()
            #classif=labels.fit(samples)
            #print classif
            print labels
            print sample
            
            labels = labels.reshape(samples.shape)

            plt.figure(figsize=(5, 5))
            plt.imshow(samples, cmap=plt.cm.gray)
            for l in range(N_REGIONS):
                plt.contour(labels == l, contours=1,
                            colors=[plt.cm.spectral(l / float(N_REGIONS))])
            plt.xticks(())
            plt.yticks(())
            title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
            print(title)
            plt.title(title)
        plt.show() 
        
    
    def _compute_comparison_scores_unlabeled(self):
        """Compute the scores, thanks to cross validation"""

        # Generate sample data
        #print"########### initial samples "
        samples = self._data
        
        print "samples before classification"
        print samples
        #print samples
        pits=samples['pit']
       
        #print pits
        columns = samples.columns.tolist()
        
        print"########### sample columns"
        print columns
        for column in ['pit', 'position', 'condition']:
            if column in columns:
                del columns[columns.index(column)] 
        #print samples
        samples = samples[columns].values
        
        print"########### sample values"
        print samples
        
        
        
        
        self._k_mean(samples, pits)
        #self._agglomerative_clustering(samples,pits,'unstructured')
        #self._spectral_clustering(samples)
        #self._dbscan(samples,pits)
        #self._various_algorithm(samples,pits)
        
        
    
    def _compute_comparison_scoress_with_cross_validation(self, key='condition'):
        """Compute the scores, thanks to cross validation"""


        samples = self._data
        

        #print"########### samples"
        #print samples

        
        # make a transformation of the labels
        labels = samples[key].values
        unique_labels = np.unique(labels)

        if len(unique_labels) == 2:
            assert len(unique_labels) == 2, 'We have not tested something else' #XXX  Need to modify the scripts for something else
            assert 'LIVING' in unique_labels
            assert 'FIXED' in unique_labels
            labels[labels=='LIVING'] = +1
            labels[labels=='FIXED'] = -1
        else:
            warnings.warn("We use several labels and it has not been deeply tested")
            print labels
        print"########### LABELS"
        print labels
        # get the columns of interest
        columns = samples.columns.tolist()
        #print"########### columns"
        #print columns
        for column in ['pit', 'position', 'condition']:
            if column in columns:
                del columns[columns.index(column)] 

        # We have few elements => use a leave one out method
        values = samples[columns].values

        
        #print"########### values"
        #print values

        scores = np.zeros(len(values))
        scores_label=np.zeros(len(values))
        loo = LeaveOneOut(len(values))
        #print loo
        #logging.info('Number of samples: ' + str(len(values)))
        logging.info('Classifier: %s' , self._selected_classifier)


        
        for train, test in loo:
            
            
            #print train 
            #print test

            # Get the partitions
            samples_train = values[train]
            samples_test = values[test].reshape(1,  - 1)
            #print samples_test
            labels_train = labels[train]
            labels_test = labels[test]
            #logging.info("label to found before running classification" + str(labels[test]))

            #logging.info("value to test to found before running classification" + str(values[test]))

            
            # Normalize data when KNN is not used (ie. metric with histo comparison)
            if self._selected_classifier not in ('KNN_CHI2', 'KNN_EUCL'):
                scaler = StandardScaler().fit(samples_train)                
                samples_train = scaler.transform(samples_train)
                samples_test = scaler.transform(samples_test)
                #samples_test = np.array(samples_test).reshape((len(samples_test)-1, 1))

            if self._selected_classifier in ('SVC-PCA', 'RF-PCA'):
                decomposition = PCA().fit(samples_train)
                samples_train = decomposition.transform(samples_train)
                samples_test = decomposition.transform(samples_test)

            # Train the classifier
            if self._selected_classifier.startswith('SVC'):
                clf = SVC(probability=True)
                clf.fit(samples_train, labels_train.tolist())
#                logging.info("SVC sample test control label :" + str(labels[test]))
#                logging.info("SVC sample test predicted label :" + str(clf.predict(samples_test)))
                
            
            elif self._selected_classifier.startswith('RF'):
                clf = RandomForestClassifier(n_jobs=-1)
                clf.fit(samples_train, labels_train.tolist())
#                logging.info("RF sample test control label :" + str(labels[test]))
#                logging.info("RF sample test predicted label :" + str(clf.predict(samples_test)))

            
            elif self._selected_classifier.startswith('KNN_EUCL'):
                clf = KNeighborsClassifier(metric='minkowski', p=2)
                clf.fit(samples_train, labels_train.tolist())
#                logging.info("KNN eucl sample test control label :" + str(labels[test]))
#                logging.info("KNN eucl  sample test predicted label :" + str(clf.predict(samples_test)))
            
            elif self._selected_classifier.startswith('KNN_CHI2'):
                #logging.info('selected_classifier.startswith(KNN_CHI2)')
                clf = KNeighborsClassifier(metric=additive_chi2_kernel)
                clf.fit(samples_train, labels_train.tolist())
#                logging.info("KNN chi2 sample test control label :" + str(labels[test]))
#                logging.info("KNN chi2  sample test predicted label :" + str(clf.predict(samples_test)))

            # make the classification
            # XXX Hack to see if it is multiclass or not
            if 2 == len(np.unique(labels_train)) :
                #logging.info('1 clf.predict_proba(samples_test)[:,-1]')
                score = clf.predict_proba(samples_test)[:,-1]
                #print clf.predict(samples_test)[:,-1]
                score_label = clf.predict(samples_test)
                #logging.info('2 clf.predict_proba(samples_test)[:,-1]')
                #print score
                #print labels_test

            else:
                score = clf.predict(samples_test)
            scores[test] = score
            scores_label[test] = score_label
            
            #counter+=1

        self._scores = scores
        self._scores_label=scores_label
        #print scores
        #print scores_label
        self._true_labels = labels



    def visualize_performance(self):

        self._common_visualize_performance(self._true_labels, self._scores)



class DensityHistoClassifier(ClassificationExperimentMachineLearning):
    """We want an experiment which keeps only the density histogram"""

    def __init__(self, data, classifier, columns,type,density_histogram_labels,msd_histogram_labels,dinst_histogram_labels):
        super(DensityHistoClassifier, self).__init__(data, "density_histo_classifier", classifier)
        print "create new density histo classifier for "+ columns
        self.columns = columns
        self.type=type
        self.density_histogram_labels=density_histogram_labels     
        self.msd_histogram_labels=msd_histogram_labels
        self.dinst_histogram_labels=dinst_histogram_labels

    
    def get_columns_to_remove(self):
        columns = self._data.columns.tolist()


        if self.columns == 'density_hist':
            keep_only = self.density_histogram_labels
            
        elif self.columns == 'diff_hist':
            keep_only = self.dinst_histogram_labels
        elif self.columns == 'trajectory_hist':
            keep_only = TRAJECTORY_FRAMES_HISTOGRAM_LABELS
            
        elif self.columns == 'msd_hist':
            keep_only = self.msd_histogram_labels
        elif self.columns == 'all':
            return []
        else:
            raise Exception(self.columns + ' unknown')


        return set(columns) - set(keep_only)

    


        

# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2013, LaBRI'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'romain.giot@u-bordeaux1.fr'
__status__ = 'Prototype'

