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
            ward = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward',
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
            for l in range(N_CLUSTERS):
                plt.contour(label == l, contours=1,
                            colors=[plt.cm.spectral(l / float(N_CLUSTERS)), ])
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

                        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=N_CLUSTERS,affinity=metric)
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

                        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=N_CLUSTERS,affinity=metric)
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

                        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=N_CLUSTERS,affinity=metric)
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