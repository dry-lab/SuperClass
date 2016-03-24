#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Quick test to analyze trajectories


Nodes are point captured.
Edge represent an movment between a point and another one
"""

# imports
import glob
import sys
import os

import pandas as pd
from tulip import *

def analyze_trajectory(fname):
    """Analyse the trajectory
     - Read the file
     - Build a tulip graph
     """
    make_tulip_graph(fname)

def make_tulip_graph(fname):
    """Create a tulip graph from the trajectories"""

    LUT_POS_TO_NODE = {}
    LUT_NODE_TO_POS = {}


    df = pd.io.parsers.read_csv(fname,
                                names = ['TRAJ_ID', 'POS_STEP', 'X', 'Y', 'UNKNW1', 'UNKNW2'],
                                sep=None,
                                skipinitialspace=True,
                                index_col=False)

    i = 0
    nb = len(df.index)

    G = tlp.newGraph()
    nodeX = G.getDoubleProperty("posX")
    nodeY = G.getDoubleProperty("posY")
    viewLayout = G.getLayoutProperty("viewLayout")

    # read each trajectory
    for trajectory, points in df.groupby('TRAJ_ID', sort='POS_STEP'):
        COL_X = points['X']
        COL_Y = points['Y']

        previous_node = None
        for pos in zip(COL_X, COL_Y):
            i = i+1

            # get the current node
            if pos not in LUT_POS_TO_NODE:
                current_node = G.addNode()
                nodeX[current_node] = pos[0]
                nodeY[current_node] = pos[1]
                viewLayout[current_node] = tlp.Coord(pos[0], pos[1], 0)
                LUT_POS_TO_NODE[pos] = current_node
            else:
                current_node = LUT_POS_TO_NODE[pos]

            # add the link (if needed)
            if previous_node:
                G.addEdge(previous_node, current_node)

            previous_node = current_node

        sys.stdout.write("\r\x1b[K"+ str(i*100/nb) + '/100')
        sys.stdout.flush()

    tlp.saveGraph(G, fname+'.tlp.gz')
    print
    return G
        
# code
if __name__ == '__main__':

    #FNAME = '../../data/31102013_08.38/Manip_31102013_08.38/B 1/P 1/SR_001.MIA/tracking/SR_001_MIA.trc'
    #analyze_trajectory(FNAME)
    for fname in glob.glob('../../data/31102013_08.38/Manip_31102013_08.38/*/*/*/tracking/*MIA.trc'):
        print 'Treat', fname
        analyze_trajectory(fname)


# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2013, LaBRI'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'romain.giot@u-bordeaux1.fr'
__status__ = 'Prototype'

