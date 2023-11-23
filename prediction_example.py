# -*- coding: utf-8 -*-

import numpy as np
from limb.data_preprocess import preprocess_data
from limb.data_preprocess import interchange_transform as limb_leads_interchange
from limb.LDenseNet import build_model as limb_model
from chest.get_dataset import interchange_transform as chest_leads_interchange
from chest.ILC_model import build_model as chest_model

def limb_leads_misplacement_detection(samples, model_save_path):
    preprocessed_samples = preprocess_data(samples, downsample = True, denoise = True)
    idx = [2,3,-1]
    preprocessed_samples = preprocessed_samples[:,idx,:]
    preprocessed_samples = np.transpose(preprocessed_samples, [0,2,1])
    model = limb_model(input_shape = (1200,3), output_dims=4)
    model.load_weights(model_save_path)
    result = model.predict(preprocessed_samples)
    result = np.argmax(result, axis=-1)
    return result
    
def chest_leads_misplacement_detection(samples, model_save_path):
    preprocessed_samples = preprocess_data(samples, downsample = True, denoise = True)
    preprocessed_samples = preprocessed_samples[:,6:,:]
    preprocessed_samples = np.transpose(preprocessed_samples, [0,2,1])
    model = chest_model(input_shape = (1200,6), output_dims=10)
    model.load_weights(model_save_path)
    result = model.predict(preprocessed_samples)
    result = np.argmax(result, axis=-1)
    return result

if __name__ == '__main__':
    
    
    samples = np.load('D:/ecg_interchange_new/code/example/samples.npy')#(10,5000,12)
    samples = np.transpose(samples, [0,2,1])#(10,12,5000)
    
    #simulate the limb misplacement situation
    # 0: Normal, 1:LA/RA, 2:RA/LL, 3:LA//LL
    limb_leads_misplacement_samples = np.copy(samples)
    mode = [0,1,2,3,0,1,2,3,0,1]
    for i in range(samples.shape[0]):
        limb_leads_misplacement_samples[i] = limb_leads_interchange(mode[i],limb_leads_misplacement_samples[i])
    
    #prediction
    model_save_path = 'D:/ecg_interchange_new/code/saved_model/LDenseNet/'
    result = limb_leads_misplacement_detection(limb_leads_misplacement_samples, model_save_path)
    print(result)
    #model ouput:
    #0 --> Normal
    #1 --> LA/RA
    #2 --> RA/LL
    #3 --> LA//LL
   
    
    #simulate the limb misplacement situation
    # 0:Normal, 1:V1/V2, 2:V1/V3, 6:V2/V3, 7:V2/V4, 10:V3/V4, 11:V3/V5, 13:V4/V5, 14:V4/V6, 15:V5/V6
    mode = [0,1,2,6,7,10,11,13,14,15]
    chest_leads_misplacement_samples = np.copy(samples)
    for i in range(samples.shape[0]):
        chest_leads_misplacement_samples[i] = chest_leads_interchange(mode[i],chest_leads_misplacement_samples[i])
    
    #prediction
    model_save_path = 'D:/ecg_interchange_new/code/saved_model/ILC_model/'
    result = chest_leads_misplacement_detection(chest_leads_misplacement_samples, model_save_path)
    print(result)
    #model ouput:
    #0 --> Normal
    #1 --> V1/V2
    #2 --> V1/V3
    #3 --> V2/V3
    #4 --> V2/V4
    #5 --> V3/V4
    #6 --> V3/V5
    #7 --> V4/V5
    #8 --> V4/V6
    #9 --> V5/V6

    
    
    
