"""
This file defines global information about MEG data, such as:

- Sessions (rest, task,...)
- Groups   (AV, V, AVr)
- Subjects
- Files' location
"""

import sys, os
from os.path import dirname, abspath


# SUBJECTS_DIR = '/neurospin/meg/meg_tmp/Dynacomp_Ciuciu_2011/yousra'
# DATASET_DIR  = '/neurospin/tmp/Omar/data'
# FSAVERAGE_DIR = '/neurospin/tmp/Omar/data'


SUBJECTS_DIR = '/neurospin/meg/meg_tmp/Dynacomp_Ciuciu_2011/yousra'
DATASET_DIR  = '/neurospin/tmp/Omar/data'
FSAVERAGE_DIR = 'C:\\Users\\omard\\Documents\\data_to_go\\dynacomp'


def get_info(dataset_dir = DATASET_DIR, 
             subjects_dir = SUBJECTS_DIR, 
             fsaverage_dir = FSAVERAGE_DIR):
    """
    Args:
        dataset_dir: path to folder containing /SSS
        subjects_dir: path to folder containing /subjects
        fsaverage_dir: path to folder containing /fsaverage

    Returns a dictionary (info) containing:
        - info['groups']
        - info['subjects']
        - info['sessions']

        - info['dataset_dir']
        - info['subjects_dir']
        - info['fsaverage_dir']
        - info['dataset_dir_output']
        - info['dataset_dir_mf_output']
        - info['paths_to_subjects']        ; example: paths_to_subjects['AV']['nc_110174']
        - info['paths_to_events']          ; example: paths_to_events['AV']['nc_110174']
        - info['paths_to_subjects_output'] ; example: paths_to_subjects_output['AV']['nc_110174']
    """
    info = {}
    #---------------------------------------------------------------------------
    # Groups and subjects
    #---------------------------------------------------------------------------
    # groups
    groups      = ['AV','V', 'AVr']
    # subjects
    subjects    = {}
    subjects['AV'] = ['nc_110174','da_110453','mb_110421','bl_110396','fp_110067','kr_080082', \
            'ks_110142','ld_110370','mp_110340','na_110353','pc_110210','pe_110338']

    subjects['V'] = ['jh_100405','gc_100388','jm_100109','vr_100551','fb_110137','aa_100234', \
       'cl_100240','jh_110224','mn_080208','in_110286','tl_110313','cm_110222']

    subjects['AVr'] = ['jm_100042','cd_100449','ap_110299','ma_130185','mj_130216','rg_110386', \
      'ga_130053','jd_110235','sa_130042','bd_120417','ak_130184','mr_080072']


    info['groups'] = groups
    info['subjects'] = subjects
    #---------------------------------------------------------------------------
    # Sessions
    #---------------------------------------------------------------------------
    sessions = ['rest0', 'rest5', 'pretest', 'posttest']

    info['sessions'] = sessions

    #---------------------------------------------------------------------------
    # Path to data
    #---------------------------------------------------------------------------
    # current folder
    current_dir = dirname(abspath(__file__))
    # project folder
    project_dir = current_dir

    # output folders

    # - output to source reconstruction on parcellation and to mf analysis of each raw file
    dataset_dir_output = os.path.join(project_dir, 'data_out') 
    # - output folder to files summarizing mf analysis
    dataset_dir_mf_output = os.path.join(project_dir, 'data_mf_out')

    # path to subjects data, example: paths_to_subjects['AV']['nc_110174']
    paths_to_subjects = {}
    for group in groups:
        paths_to_subjects[group] = {}
        for subject in subjects[group]:
                paths_to_subjects[group][subject] = \
                    os.path.join(dataset_dir, 'SSS', group,subject)


    # path to events data, example: paths_to_events['AV']['nc_110174']
    paths_to_events = {}
    for group in groups:
        paths_to_events[group] = {}
        for subject in subjects[group]:
                paths_to_events[group][subject] = \
                    os.path.join(dataset_dir, 'SSS', group, subject,'events','raw')


    # path to subjects output data, example: paths_to_subjects_output['AV']['nc_110174']
    paths_to_subjects_output = {}
    for group in groups:
        paths_to_subjects_output[group] = {}
        for subject in subjects[group]:
                paths_to_subjects_output[group][subject] = \
                    os.path.join(dataset_dir_output, 'source_rec_on_parcel', group,subject)


    info['dataset_dir'] = dataset_dir
    info['subjects_dir'] = subjects_dir
    info['fsaverage_dir'] = fsaverage_dir
    info['dataset_dir_output'] = dataset_dir_output
    info['dataset_dir_mf_output'] = dataset_dir_mf_output
    info['paths_to_subjects'] = paths_to_subjects
    info['paths_to_events'] = paths_to_events
    info['paths_to_subjects_output'] = paths_to_subjects_output

    return info
