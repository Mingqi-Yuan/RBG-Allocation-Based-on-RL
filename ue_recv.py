# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:16:03 2019

@author: tao
"""

from scipy import io
import numpy as np

MCS_INFO = dict()

def load_mcs_info():
    global MCS_INFO
    mcs_file = r'mcs_bler-mcs1~14frame=500,15~29frame=100.mat'
    
    mat_mcs_info = io.loadmat(mcs_file)['MCS']
    
    for mcs_idx in range(len(mat_mcs_info)):
        mcs = mcs_idx + 1
        MCS_INFO[mcs] = dict()
    
        mcs_info = mat_mcs_info[mcs-1]
        while len(mcs_info) == 1:
            mcs_info = mcs_info[0]
        MCS_INFO[mcs]['modulation'] = mcs_info[0][0]
        MCS_INFO[mcs]['code_rate'] = mcs_info[1][0][0]
        MCS_INFO[mcs]['spectral_efficiency'] = mcs_info[2][0][0]
        
        MCS_INFO[mcs]['bler'] = dict()
        for bler_info in mcs_info[3]:
            while len(bler_info) == 1:
                bler_info = bler_info[0]
            tbs = bler_info[0]
            while type(tbs) == np.ndarray:
                tbs = tbs[0]
            snr_bler = dict(zip(bler_info[1][0], bler_info[1][1]))
            MCS_INFO[mcs]['bler'][tbs] = snr_bler
            
def get_se(mcs):
    mcs = min(29, max(1, mcs))
    if len(MCS_INFO) == 0:
        load_mcs_info()
    return MCS_INFO[mcs]['spectral_efficiency']

def get_snr_bler(mcs, tbs): 
    if len(MCS_INFO) == 0:
        load_mcs_info()
    tbs_snr_bler = MCS_INFO[mcs]['bler']
    ref_tbses = np.int32(list(tbs_snr_bler.keys()))
    ref_tbs = list(tbs_snr_bler.keys())[np.argmin(np.abs(ref_tbses-tbs))]
    snr_bler = tbs_snr_bler[ref_tbs]

    return snr_bler

def get_bler(mcs, tbs, snr):
    print('snr is', snr)
    snr_bler = get_snr_bler(mcs, tbs)
    
    ref_snrs, ref_blers = np.float64(list(snr_bler.keys())), np.float64(list(snr_bler.values()))
    if snr < np.min(ref_snrs):
        bler = snr_bler[np.min(ref_snrs)]
    elif snr > np.max(ref_snrs):
        bler = snr_bler[np.max(ref_snrs)]
    else:
        ref_snr_idx = np.argsort(np.abs(ref_snrs-snr))[:2]
        ref_snr = ref_snrs[ref_snr_idx]
        ref_bler = ref_blers[ref_snr_idx]
        bler = np.sum(np.abs(ref_snr-snr)[::-1] / np.abs(ref_snr[0]-ref_snr[1]) * ref_bler)
        
    return bler

def is_ack(mcs, tbs, snr):
    bler = get_bler(mcs, tbs, snr)
    rand = np.random.rand()
    if rand < bler:
        return False
    else:
        return True

