# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:07:26 2022

@author: nyapass
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import neurokit2 as nk
import scipy.signal


# butterworth bandpass filter
def bandpass(x):
    '''
    x: input signal, numpy array with shape (length,)
    '''
    y = nk.signal_filter(x, sampling_rate=500, lowcut=0.5, highcut=35, order=5)
    return y

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


# preprocess one sample
def leads_preprocessing(leads, downsample=True, denoise = True):
    '''
    leads: input sample, numpy array with shape (channel, length)
    '''
    pre_lead = []
    for i in range(leads.shape[0]):
        lead = leads[i]
        if denoise:
            lead = bandpass(lead)
        if downsample:
            lead = scipy.signal.resample(lead, int(lead.shape[0]/500*120))
        lead = ZscoreNormalization(lead)
        pre_lead.append(lead)
    pre_lead = np.array(pre_lead, dtype = 'float32')
    return pre_lead

# simulation the limb lead misplacement situations 
def interchange_transform(mode,normal_leads):
    '''
    mode: index for limb lead misplacement situations int
    normal_leads: sample without misplacment, numpy array with shape (channel, length)
    '''
    lead_num = normal_leads.shape[0]
    if lead_num==6:
        transformed_leads = np.copy(normal_leads)
        if mode == 1:
            transformed_leads[0]=normal_leads[1]
            transformed_leads[1]=normal_leads[0]
        elif mode == 2:
            transformed_leads[0]=normal_leads[2]
            transformed_leads[2]=normal_leads[0]
        elif mode == 3:
            transformed_leads[0]=normal_leads[3]
            transformed_leads[3]=normal_leads[0]
        elif mode == 4:
            transformed_leads[0]=normal_leads[4]
            transformed_leads[4]=normal_leads[0]
        elif mode == 5:
            transformed_leads[0]=normal_leads[5]
            transformed_leads[5]=normal_leads[0]
        elif mode == 6:
            transformed_leads[1]=normal_leads[2]
            transformed_leads[2]=normal_leads[1]
        elif mode == 7:
            transformed_leads[1]=normal_leads[3]
            transformed_leads[3]=normal_leads[1]
        elif mode == 8:
            transformed_leads[1]=normal_leads[4]
            transformed_leads[4]=normal_leads[1]
        elif mode == 9:
            transformed_leads[1]=normal_leads[5]
            transformed_leads[5]=normal_leads[1]
        elif mode == 10:
            transformed_leads[2]=normal_leads[3]
            transformed_leads[3]=normal_leads[2]
        elif mode == 11:
            transformed_leads[2]=normal_leads[4]
            transformed_leads[4]=normal_leads[2]
        elif mode == 12:
            transformed_leads[2]=normal_leads[5]
            transformed_leads[5]=normal_leads[2]
        elif mode == 13:
            transformed_leads[3]=normal_leads[4]
            transformed_leads[4]=normal_leads[3]
        elif mode == 14:
            transformed_leads[3]=normal_leads[5]
            transformed_leads[5]=normal_leads[3]
        elif mode == 15:
            transformed_leads[4]=normal_leads[5]
            transformed_leads[5]=normal_leads[4]
        else:
            pass
    else:
        transformed_leads = np.copy(normal_leads)
        if mode == 1:
            transformed_leads[6]=normal_leads[7]
            transformed_leads[7]=normal_leads[6]
        elif mode == 2:
            transformed_leads[0+6]=normal_leads[2+6]
            transformed_leads[2+6]=normal_leads[0+6]
        elif mode == 3:
            transformed_leads[0+6]=normal_leads[3+6]
            transformed_leads[3+6]=normal_leads[0+6]
        elif mode == 4:
            transformed_leads[0+6]=normal_leads[4+6]
            transformed_leads[4+6]=normal_leads[0+6]
        elif mode == 5:
            transformed_leads[0+6]=normal_leads[5+6]
            transformed_leads[5+6]=normal_leads[0+6]
        elif mode == 6:
            transformed_leads[1+6]=normal_leads[2+6]
            transformed_leads[2+6]=normal_leads[1+6]
        elif mode == 7:
            transformed_leads[1+6]=normal_leads[3+6]
            transformed_leads[3+6]=normal_leads[1+6]
        elif mode == 8:
            transformed_leads[1+6]=normal_leads[4+6]
            transformed_leads[4+6]=normal_leads[1+6]
        elif mode == 9:
            transformed_leads[1+6]=normal_leads[5+6]
            transformed_leads[5+6]=normal_leads[1+6]
        elif mode == 10:
            transformed_leads[2+6]=normal_leads[3+6]
            transformed_leads[3+6]=normal_leads[2+6]
        elif mode == 11:
            transformed_leads[2+6]=normal_leads[4+6]
            transformed_leads[4+6]=normal_leads[2+6]
        elif mode == 12:
            transformed_leads[2+6]=normal_leads[5+6]
            transformed_leads[5+6]=normal_leads[2+6]
        elif mode == 13:
            transformed_leads[3+6]=normal_leads[4+6]
            transformed_leads[4+6]=normal_leads[3+6]
        elif mode == 14:
            transformed_leads[3+6]=normal_leads[5+6]
            transformed_leads[5+6]=normal_leads[3+6]
        elif mode == 15:
            transformed_leads[4+6]=normal_leads[5+6]
            transformed_leads[5+6]=normal_leads[4+6]
        else:
            pass
    return transformed_leads

#generarte label according to appointed misplacement situations and rate for each class
def make_label(all_data_num, positive_list, positive_rate):
    '''
    all_data_num: number of all dataset, int
    positive_list: list of appointed misplacement situations, list
    positive_rate: list of rate for each class, list
    '''
    labels = []
    for i in range(len(positive_rate)):
        kind_num = int(positive_rate[i]*all_data_num)
        for j in range(kind_num):
            labels.append(positive_list[i])
    while len(labels) < all_data_num:
        labels.append(0)
    labels = np.array(labels)
    index = np.arange(labels.shape[0])
    np.random.seed(1096)
    np.random.shuffle(index)
    labels = labels[index]
    return labels

#split training set and validaion set
def train_val_split(raw_data, val_rate):
    '''
    raw_data: dataset, numpy array with shape (sample_number, channel, length)
    val_rate: the rate for validation set, float
    '''
    index = np.arange(raw_data.shape[0])
    val_lens = int(raw_data.shape[0]*val_rate)
    np.random.seed(1095)
    val_index = np.random.choice(index, val_lens, replace = False)
    train_index = []
    for idx in index:
        if idx not in val_index:
            train_index.append(idx)
    train_data = raw_data[train_index]
    val_data = raw_data[val_index]
    return train_data, val_data

#prepare for the datasets
def split_data(pre_data, val_rate, positive_list, positive_rate_test, fix = False, otherlist = [1,2,4,5]):
    '''
    pre_data: preprocessed data, numpy array with shape (sample_number, channel, length)
    val_rate: validation rate, float
    positive_list: list of appointed misplacement situations, list
    positive_rate_test: list of rate for each class in validation set and test set, list
    fix: used when compare the binary classification performance with machine learning-based method, bool
    otherlist: only used when fix is True. list of appointed misplacement situations, list
    '''
    def shuffle_train_data(x,y):
        index = np.arange(x.shape[0])
        np.random.seed(1096)
        np.random.shuffle(index)
        x = x[index]
        y = y[index]
        return x, y
    
    def second_split(labels, ol):
        def binary_label(labels):
            new_labels = []
            for i in range(labels.shape[0]):
                if labels[i]!=0:
                    new_labels.append(1)
                else:
                    new_labels.append(0)
            new_labels = np.array(new_labels)
            return new_labels
        labels = binary_label(labels)
        one_position = np.where(labels==1)[0]
        other_num = int(len(one_position)/len(ol))
        other_label = []
        for i in ol:
            for j in range(other_num):
                other_label.append(i)
        while len(other_label) < len(one_position):
            other_label.append(1)
        count = 0
        for k in range(labels.shape[0]):
            if labels[k]==1:
                labels[k] = other_label[count]
                count+=1
        np.random.seed(1098)
        np.random.shuffle(labels)
        return labels
                
    train_data, val_data = train_val_split(pre_data, val_rate) 
    #随机性取决于 原数据总长 和 val rate 
    train_data_list = []
    train_label_list = []
    for i in range(len(positive_list)):
        train_data_list.append(train_data)
        train_label_list.append(np.full((train_data.shape[0]), positive_list[i]))
    extend_train_data = np.concatenate(train_data_list, axis=0)
    extend_train_label = np.hstack(train_label_list)
    val_labels = make_label(val_data.shape[0], positive_list, positive_rate_test)
    # 随机性取决于 positive_rate_test， 原数据总长
    if fix:
        val_labels = make_label(val_data.shape[0], [0,1], [0.5, 0.5])
        val_labels = second_split(val_labels, otherlist)
        
    x_train = []
    for j in range(extend_train_data.shape[0]):
        mode = extend_train_label[j]
        t_leads = interchange_transform(mode, extend_train_data[j])
        x_train.append(t_leads)
        
    x_val = []
    for k in range(val_data.shape[0]):
        mode = val_labels[k]
        t_leads = interchange_transform(mode, val_data[k])
        x_val.append(t_leads)
          
    x_train = np.array(x_train, dtype = 'float32')
    x_val = np.array(x_val, dtype = 'float32') 

    x_train, y_train = shuffle_train_data(x_train, extend_train_label)
    return x_train, y_train, x_val, val_labels


                    
def preprocess_data(raw_data, downsample = True, denoise = True):
    '''
    raw_data: dataset, numpy array with shape (sample_number, channel, length)
    '''
    if raw_data.shape[-1] < raw_data.shape[1]:
        raw_data = np.transpose(raw_data, [0,2,1])
    preprocessed_data = []
    for i in range(raw_data.shape[0]):
        print(i/raw_data.shape[0], i)
        ecgs = leads_preprocessing(raw_data[i], downsample, denoise)
        preprocessed_data.append(ecgs)
    preprocessed_data = np.array(preprocessed_data, dtype = 'float32')
    return preprocessed_data

if __name__ == '__main__':
    
    # training data and validation data
    '''
    preprocessed_data = np.load("path of preprocessed data of Chapman-shaoxing")#(N, 12, 1200)
    preprocessed_data = preprocessed_data[:,6:,:]
    val_rate = 0.2
    positive_list = [0,1,2,6,7,10,11,13,14,15]
    positive_rate_test = [0.5,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9]
    x_train, y_train, x_val, y_val = split_data(preprocessed_data, val_rate, positive_list, positive_rate_test)
    np.save('.../x_train.npy', x_train)
    np.save('.../y_train.npy', y_train)
    np.save('.../x_val.npy', x_val)
    np.save('.../y_val.npy', y_val)
    print(Counter(y_train))
    print(Counter(y_val))
    '''
    
    # PTBXL testing data
    '''
    preprocessed_data = np.load("path of preprocessed data of ptbxl")#(N, 12, 1200)
    preprocessed_data = preprocessed_data[:,6:,:]
    val_rate = 1.0
    positive_list = [0,1,2,6,7,10,11,13,14,15]
    positive_rate_test = [0.5,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9]
    x_train, y_train, x_val, y_val = split_data(preprocessed_data, val_rate, positive_list, positive_rate_test)
    np.save('.../x_test.npy', x_val)
    np.save('.../y_test.npy', y_val)
    print(Counter(y_train))
    print(Counter(y_val))
    '''
    
    # PTB testing data
    '''
    preprocessed_data = np.load("path of preprocessed data of ptb")#(N, 12, 1200)
    preprocessed_data = preprocessed_data[:,6:,:]
    val_rate = 1.0
    positive_list = [0,1,2,6,7,10,11,13,14,15]
    positive_rate_test = [0.5,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9,0.5/9]
    x_train, y_train, x_val, y_val = split_data(preprocessed_data, val_rate, positive_list, positive_rate_test)
    np.save('.../x_test.npy', x_val)
    np.save('.../y_test.npy', y_val)
    print(Counter(y_train))
    print(Counter(y_val))
    '''
    
   