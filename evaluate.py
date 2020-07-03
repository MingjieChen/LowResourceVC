'''
    This is a script for objective evaluation on vcc2018 parallel evaluation dataset.
    This script calcuate :
        1. mcd: dtw aligned mel cepstral distortion between converted and target speech mcep.
        2. msd: log modulation spectra distance between converted and target speech mcep.
'''
import argparse
import mcd.metrics as mt
import os
import glob
import numpy as np

from mcd import dtw
from utils import world_encode_wav
from tqdm import tqdm
from os.path import exists, dirname, basename, join
import os
from scipy.fftpack import fft

def msd(cvt, trg):
    
    cvt_ms = np.log(np.abs(fft(cvt.T, n = 64)))
    trg_ms = np.log(np.abs(fft(trg.T, n = 64)))
    
    msd = np.sqrt(np.mean((cvt_ms - trg_ms)**2))
    return msd

def dtw_mcd(cvt, trg):
    
    cvt_mcep = cvt.astype('float64')
    trg_mcep = trg.astype('float64')
    cvt_mcep = cvt_mcep[:,1:]
    trg_mcep = trg_mcep[:, 1:]
    '''
    for t, c in zip(trg_mcep, cvt_mcep):
        cost, _ = dtw.dtw(t, c, mt.logSpecDbDist)
        
        frames = len(trg_mcep)
        total_cost += cost
        total_frames += frames
    '''

    cost, _ = dtw.dtw(trg_mcep, cvt_mcep, mt.logSpecDbDist)
    return cost / len(trg_mcep)



def extract_mcep(config):
    # extract mcep for converted and reference
    # record mcep paths for cvt and ref
    with open(config.pair_list_path, 'r') as f:
        pair_list = f.readlines()
    
    print(f"load in {len(pair_list)} pairs")
    
    os.makedirs(config.mcep_tmp_path, exist_ok = True)

    mcep_pair_list = []
    for l in pair_list:
        cvt_path, ref_path = l.strip().split()
        
        if not exists(cvt_path):
            raise Exception(f"cvt {cvt_path} does not exist")
        if not exists(ref_path):
            raise Exception(f"ref {ref_path} does not exist")
        
        wav_id = basename(ref_path).split('.')[0]
        trg = basename(dirname(ref_path))
        cvt = basename(cvt_path).split('.')[0]
        

               
        cvt_mcep_path = join(config.mcep_tmp_path, cvt+'.npy')
        ref_mcep_path = join(config.mcep_tmp_path, trg+'-'+wav_id+'.npy')
        
        #mcep_pair_list.append((cvt_mcep_path, ref_mcep_path))
        #print(f"cvt {cvt_mcep_path} ref {ref_mcep_path}", flush=True)
        #if not exists(cvt_mcep_path):
        cvt_f0, _, _, _, cvt_coded_sp = world_encode_wav(cvt_path, config.sample_rate, 36)
        #normed_cvt_sp = (cvt_coded_sp - mc_mean) / mc_std
        #np.save(cvt_mcep_path, normed_cvt_sp)
        #np.save(cvt_mcep_path, cvt_coded_sp)
        #if not exists(ref_mcep_path):
        ref_f0, _, _, _, ref_coded_sp = world_encode_wav(ref_path, config.sample_rate, 36)
        #normed_ref_sp = (ref_coded_sp - mc_mean) / mc_std
        #np.save(ref_mcep_path, normed_ref_sp)
        #np.save(ref_mcep_path, ref_coded_sp)
        
        
        _mcd = dtw_mcd(cvt_coded_sp, ref_coded_sp)
        _msd = msd(cvt_coded_sp, ref_coded_sp)
        #total_mcd.append(_mcd)
        print(f'cvt {cvt_mcep_path} ref {ref_mcep_path} \t mcd : {_mcd} \t msd: {_msd}', flush=True)

def calculate_mcd(mcep_pair_list):
    
    total_mcd = []
    for cvt, ref in mcep_pair_list:
        _mcd = dtw_mcd(cvt, ref)
        total_mcd.append(_mcd)
        print(f'cvt {cvt} ref {ref} \t mcd : {_mcd}', flush=True)
    
    print(f"final mcd {np.mean(total_mcd)}", flush=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_rate', type=int, default = 22050)
    parser.add_argument('--convert_dir', type=str, required = True)
    parser.add_argument('--speaker_path', type=str, default = './speaker_used.json')
    parser.add_argument('--pair_list_path', type=str, default = './pair_list.txt')
    parser.add_argument('--mcep_tmp_path', type=str, default = './pair_list.txt')

    config = parser.parse_args()

    mcep_list = extract_mcep(config)
    #calculate_mcd(mcep_list)

