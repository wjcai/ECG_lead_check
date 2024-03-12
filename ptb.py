
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 09:09:50 2022

@author: nyapass
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk

# Z-score normalization
def ZscoreNormalization(x):
    '''
    x: input signal, numpy array with shape (length,)
    '''
    if np.std(x)!=0:
        x = (x - np.mean(x)) / np.std(x)
    else:
        x = np.zeros_like(x)
    return x

# butterworth bandpass filter
def bandpass(x):
    '''
    x: input signal, numpy array with shape (length,)
    '''
    y = nk.signal_filter(x, sampling_rate=1000, lowcut=0.5, highcut=35, order=5)
    return y

#split the length-varied signal
def split_signal(signal_c_120):
    '''
    signal_c_120: preprocessed record, numpy array with shape (12, length)
    '''
    signal_c_120 = np.transpose(signal_c_120, [1,0])
    start = 0
    all_sample=[]
    signal_lens = int(10*120)
    while start+signal_lens <= signal_c_120.shape[0]:
        sample = signal_c_120[start:start+signal_lens]
        all_sample.append(sample)
        start+=signal_lens
    #all_sample = np.array(all_sample)
    return all_sample

# load data and preprocess
def load_raw_data(path):
    '''
    path: path of data, string
    '''
    all_data=[]
    for line in open(path+"RECORDS_e"): #exclude 5 recording manually
        data = wfdb.rdrecord(path + line[:-1])
        signal_15 = np.transpose(data.p_signal, [1,0])  #(15,xxx)
        
        if signal_15.shape[0]!=15:
            print(line)
            continue
        else:
            singnal_c = signal_15[:12]
            signal_c_120 = []
            for i in range(len(singnal_c)):
                lead = singnal_c[i]
                if np.std(lead)==0:
                    print(line)
                lead = bandpass(lead)
                lead = signal.resample(lead, int(lead.shape[0]/1000*120))
                lead = ZscoreNormalization(lead)
                signal_c_120.append(lead)
            signal_c_120 = np.array(signal_c_120)
            all_data = all_data + split_signal(signal_c_120)

    all_data = np.array(all_data)
    return all_data





    
if __name__ == '__main__':
    
    path = '.../ptb-diagnostic-ecg-database-1.0.0/' #path of data
    raw_data = load_raw_data(path) #(sample, 5000, 6)
    raw_data=np.transpose(raw_data, [0,2,1]) 
    
    np.save(".../ptb_preprocess.npy", raw_data) #path of preprocessed data
    
   