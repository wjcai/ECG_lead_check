import pandas as pd
import numpy as np
import wfdb
import ast

#load raw data
def load_raw_data(df, sampling_rate, path):
    '''
    df: dataframe of ptbxl
    sampling_rate: choose the samle rate of signal, 100 or 500
    path: root_path of data
    '''
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


if __name__ == '__main__':
    
    path = 'path of data'
    sampling_rate=500
    # exclude the pacemaker records manually
    Y = pd.read_csv(path+'ptbxl_database_withoutpacemaker.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = load_raw_data(Y, sampling_rate, path)
    np.save("raw data path of ptbxl",X)
        
