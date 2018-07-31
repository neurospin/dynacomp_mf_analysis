import os, sys
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mne.viz import plot_topomap
from copy import deepcopy, copy
from matplotlib.colors import LinearSegmentedColormap
import meg_info


def plot_brain(data, fmin = -1.0, fmax = 1.0, png_filename = None, positive_only = False):
    """
    data: array of shape (138,) to be plotted
    """

    # Load info
    info   = meg_info.get_info()
    subjects_dir = info['fsaverage_dir']

    #===========================================================================
    # MNE config
    #===========================================================================
    nb_vertices = 10242
    vertices = [np.arange(nb_vertices),np.arange(nb_vertices)] # lh, rh
    SUBJECTS_DIR = info['fsaverage_dir']
    SUBJECT = 'fsaverage'
    Parcellation = 'aparc.split-small'

    # Set SUBJECTS_DIR in the environment
    os.environ["SUBJECTS_DIR"] = SUBJECTS_DIR
    labels = mne.read_labels_from_annot(os.path.join(SUBJECTS_DIR,SUBJECT),
                                        parc= Parcellation)
    # Removes the label 'unknown'
    print('last label removed: %s'%labels.pop(-1))


    #===========================================================================
    # Create STC
    #===========================================================================
    cortex_data= np.zeros((nb_vertices*2,1))
    for k in range(len(labels)):
        if '-lh' in labels[k].name: # left hemisphere
            vert = labels[k].vertices[labels[k].vertices<nb_vertices]
        elif '-rh' in labels[k].name: # right hemisphere
            vert= nb_vertices + labels[k].vertices[labels[k].vertices<nb_vertices]

        cortex_data[vert,:] = data[k]

    # create stc
    stc = mne.SourceEstimate(cortex_data, vertices, tmin=0, tstep=1)

    #===========================================================================
    # Plot
    #===========================================================================

    fmid = (fmin + fmax)/2.0

    stc.subject='fsaverage'
    pick="Radial"
    t=0.

    if not positive_only:
        cm = mne.viz.mne_analyze_colormap([5, 10, 15], format = 'mayavi')
        # cm = get_cm()
    else:
        colors = [(105./255., 105./255.,105./255.),
                  (105./255., 105./255.,105./255.),
                  (255./255.,0.,0),
                  (255./255.,255./255.,0),
                  (255./255.,255./255.,0)]
        cm_name='my_cm'
        cm = LinearSegmentedColormap.from_list(cm_name,colors, N=256)
        cm = 255.*np.array([cm(ind) for ind in np.arange(256)])


    brain = stc.plot(hemi = 'split', views = ['lat','med'],
            colormap= cm,subjects_dir=subjects_dir)

    # try:
    if True:
        if not positive_only:
            brain.scale_data_colormap(fmin=fmin, fmid=fmid, fmax=fmax, transparent=False)
            # Set transparency around 0
            for h in ['lh', 'rh']:
                data = brain.data_dict[h]
                table = data["orig_ctable"].copy()
                n_colors = table.shape[0]
                n_colors2 = int(n_colors / 2)
                table[:n_colors2, -1] = np.linspace(255, 0, n_colors2) # alpha values
                table[n_colors2:, -1] = np.linspace(0, 255, n_colors2)
                brain.data_dict[h]["orig_ctable"]=table
                brain.data_dict[h]["transparent"] # set information
                for s,surf in enumerate(data['surfaces']): # indicate new colormap
                    brain.data_dict[h]['surfaces'][s].module_manager.scalar_lut_manager.lut.table = table
        else:
            brain.scale_data_colormap(fmin=fmin, fmid=fmid, fmax=fmax, transparent=True)
    # except:
    #     pass

    # Save figure
    if png_filename is not None:
        brain.save_image(png_filename)
        brain.close()

    return labels, cm


# def get_cm():
    
#     cdict2 = {'red':   ((0.0, 0.0, 0.0),
#                    (0.5, 0.0, 1.0),
#                    (1.0, 0.1, 1.0)),

#          'green': ((0.0, 0.0, 0.0),
#                    (1.0, 0.0, 0.0)),

#          'blue':  ((0.0, 0.0, 0.1),
#                    (0.5, 1.0, 0.0),
#                    (1.0, 0.0, 0.0))
#         }

#     blue_red2 = LinearSegmentedColormap('BlueRed2', cdict2)

#     return plt.cm.viridis