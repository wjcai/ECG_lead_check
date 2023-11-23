# -*- coding: utf-8 -*-

import numpy as np
import neurokit2 as nk
import scipy.signal


def bandpass(x):
    y = nk.signal_filter(x, sampling_rate=500, lowcut=0.5, highcut=35, order=5)
    return y


def ZscoreNormalization(x):
    if np.std(x)!=0:
        x = (x - np.mean(x)) / np.std(x)
    else:
        x = np.zeros_like(x)
    return x


def leads_preprocessing(leads, downsample=True, denoise = True):
    pre_lead = []
    flag=0
    for i in range(leads.shape[0]):
        lead = leads[i]
        if np.std(lead)==0:
            flag=1
        if denoise:
            lead = bandpass(lead)
        if downsample:
            lead = scipy.signal.resample(lead, int(lead.shape[0]/500*120))
        lead = ZscoreNormalization(lead)
        pre_lead.append(lead)
    pre_lead = np.array(pre_lead, dtype = 'float32')
    return pre_lead, flag


              
def preprocess_data(raw_data, downsample = True, denoise = True):
    del_list = []
    if raw_data.shape[-1] < raw_data.shape[1]:
        raw_data = np.transpose(raw_data, [0,2,1])
    preprocessed_data = []
    for i in range(raw_data.shape[0]):
        #print(i/raw_data.shape[0], i)
        ecgs, flag = leads_preprocessing(raw_data[i], downsample, denoise)
        if flag == 1:
            del_list.append(i)
        preprocessed_data.append(ecgs)
    preprocessed_data = np.array(preprocessed_data, dtype = 'float32')
    preprocessed_data = np.delete(preprocessed_data, del_list, axis=0)
    return preprocessed_data


def interchange_transform(mode,normal_leads):
    transformed_leads = np.copy(normal_leads)
    if mode == 1:#LA/RA
        transformed_leads[0]=-normal_leads[0]
        transformed_leads[1]=normal_leads[2]
        transformed_leads[2]=normal_leads[1]
        transformed_leads[3]=normal_leads[4]
        transformed_leads[4]=normal_leads[3]
    elif mode == 2:#RA/LL
        transformed_leads[0]=-normal_leads[2]
        transformed_leads[1]=-normal_leads[1]
        transformed_leads[2]=-normal_leads[0]
        transformed_leads[3]=normal_leads[5]
        transformed_leads[5]=normal_leads[3]
    elif mode == 3:#LA//LL
       transformed_leads[0]=normal_leads[1]
       transformed_leads[1]=normal_leads[0]
       transformed_leads[2]=-normal_leads[2]
       transformed_leads[4]=normal_leads[5]
       transformed_leads[5]=normal_leads[4]
    elif mode == 4:#RA>LA>LL>RA
        transformed_leads[0]=normal_leads[2]
        transformed_leads[1]=-normal_leads[0]
        transformed_leads[2]=-normal_leads[1]
        transformed_leads[3]=normal_leads[4]
        transformed_leads[4]=normal_leads[5]
        transformed_leads[5]=normal_leads[3]
    elif mode == 5:#RA>LL>LA>RA
        transformed_leads[0]=-normal_leads[1]
        transformed_leads[1]=-normal_leads[2]
        transformed_leads[2]=normal_leads[0]
        transformed_leads[3]=normal_leads[5]
        transformed_leads[4]=normal_leads[3]
        transformed_leads[5]=normal_leads[4]
    else:
        pass
    return transformed_leads


def make_label(all_data_num, positive_list, positive_rate):
    
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


def train_val_split(raw_data, val_rate):
    
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

def split_data(pre_data, val_rate, positive_list, positive_rate_test, fix = False, otherlist = [1,2,4,5]):
   
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

    train_data_list = []
    train_label_list = []
    for i in range(len(positive_list)):
        train_data_list.append(train_data)
        train_label_list.append(np.full((train_data.shape[0]), positive_list[i]))
    extend_train_data = np.concatenate(train_data_list, axis=0)
    extend_train_label = np.hstack(train_label_list)
    val_labels = make_label(val_data.shape[0], positive_list, positive_rate_test)

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

if __name__ == '__main__':
    

    #shaoxing
    '''
    raw_data = np.load("raw data path of chapman-shaoxing")
    preprocessed_data = preprocess_data(raw_data, downsample = True, denoise = True)
    np.save("path of preprocessed data of Chapman-shaoxing", preprocessed_data)
    '''
  
    #ptbxl
    '''
    raw_data = np.load("raw data path of ptbxl")
    preprocessed_data = preprocess_data(raw_data, downsample = True, denoise = True)
    preprocessed_data = np.delete(preprocessed_data, [12558], axis=0) #sample with flatten lead
    np.save("path of preprocessed data of ptbxl", preprocessed_data)
    '''
    
    #ptb
    #to ptb.py
    

    
   