# -*- coding: utf-8 -*-

import wfdb
import pathlib
import numpy as np



def ZscoreNormalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


def rn(leads):
    leads_ = np.copy(leads)
    flag=0
    leads_ = np.transpose(leads_)
    for i in range(leads_.shape[0]):
        leads_[i]=ZscoreNormalization(leads_[i])
        for j in range(4):
            if np.isnan(leads_[i][j]):
                flag = 1
                break
    return flag

def load_raw_data(path):
    all_data=[]
    error = []
    for line in open(path+"RECORDS"):
        name = path + line
        filepath = list(pathlib.Path(name[:-1]).glob('*.hea'))
        for i in filepath:
            i=str(i)
            try:
                data = wfdb.rdrecord(i[:-4])
                a=data.p_signal
                if rn(a)!=1:
                    all_data.append(a)
                else:
                    print("drop")
                    print(i)
            except Exception:
                error.append(i)
    all_data = np.array(all_data)
    all_data = np.transpose(all_data, [0,2,1])
    return all_data,error

if __name__ == '__main__':
    
    path = "path of data" 
    signals, error = load_raw_data(path)
    np.save("raw data path of chapman-shaoxing", signals)
