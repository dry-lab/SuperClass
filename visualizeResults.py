__author__ = 'mmluqman'

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
import string


def visualize_results_plate_class_color():
    nrows = 8
    ncols = 12

    image = np.zeros(nrows*ncols)

    # Set every well to a random number (color)
    # (this would be the class labels)
    image[:] = np.random.random(nrows*ncols)

    # Reshape things into a 8x12 grid
    image = image.reshape((nrows, ncols))

    col_labels = range(1,ncols+1)
    row_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    plt.matshow(image)
    plt.xticks(range(ncols), col_labels)
    plt.yticks(range(nrows), row_labels)
    plt.suptitle("Visualize classification results - Colored Plate")

    return plt


def visualize_results_plate_class_membership(data, fmt='{:.2f}', bkg_colors=['cyan', 'white']):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on a checker board pattern
        #idx = [j % 2, (j + 1) % 2][i % 2]
        #color = bkg_colors[idx]

        if (val > 0.5):
            color = bkg_colors[0]
        else:
            color = bkg_colors[1]

        tb.add_cell(i, j, width, height, text=fmt.format(val), loc='center', facecolor=color)

    # axis labels for rows ...
    for i, label in enumerate(string.ascii_uppercase[:nrows:]):
        tb.add_cell(i, -1, width, height, text=label, loc='right', edgecolor='none', facecolor='none')

    # axis labels for cols ...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label+1, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(tb)

    fig.suptitle("Visualize classification results - Plate with class memberships")

    return fig


