# -*- coding: utf-8 -*-

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from LDenseNet import build_model
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix 
import matplotlib.pyplot as plt
from data_preprocess import *


def cal_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        ALL = np.sum(confusion_matrix)
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = ALL - TP - FP - FN
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*precision*recall/(precision+recall)
        specificity = TN/(TN+FP)
        metrics_result.append([precision, recall, specificity, f1])
    return metrics_result



def evaluater(x_test, y_test,model, path):
    y_pred = model.predict(x_test)
    num = y_pred.shape[-1]
    y_pred = np.argmax(y_pred, axis=1)
    acc = len(np.where(y_pred==y_test)[0])/y_pred.shape[0]
    C = confusion_matrix(y_test, y_pred, labels=range(num))
    #np.save("C:/Users/nyapass/Desktop/papers/comfuse/ptbxl_limb.npy", C)
    plt.figure(figsize=(3.3,3.3), dpi=600)
    font = {'family': 'Arial', 'size': 10}
    plt.rc('font', **font)
    plt.matshow(C, cmap=plt.cm.Reds) 

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    name = path.split('/')[-1].split('.')[0]
    if name == 'PTBXL':
        name = '(a) ' + name
    else:
        name = '(b) ' + name
    plt.ylabel('True label')
    plt.xlabel('Predicted label' + '\n' + '\n' + name)
    #plt.savefig(path, dpi=600, format = 'tiff')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=600, format = 'tiff')

    metrics_result = cal_metrics(C)
    metrics_result = np.array(metrics_result)
    return metrics_result, acc

def merge_label(labels, to_cat=True):
    new_labels = []
    for i in range(labels.shape[0]):
        if labels[i]!=0:
            new_labels.append(1)
        else:
            new_labels.append(0)
    new_labels = np.array(new_labels)
    if to_cat:
        new_labels = to_categorical(new_labels, num_classes = 2)
    return new_labels

def evaluater_binary(x_test, y_test, model):
    y_pred = model.predict(x_test)
    num = 2
    
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = merge_label(y_pred, to_cat=False)
    y_test = merge_label(y_test, to_cat=False)
    acc = len(np.where(y_pred==y_test)[0])/y_pred.shape[0]
    
    C = confusion_matrix(y_test, y_pred, labels=range(num))
    plt.figure(figsize=(3.3,3.3), dpi=600)
    font = {'family': 'Arial', 'size': 12}
    plt.rc('font', **font)
    plt.matshow(C, cmap=plt.cm.Reds)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    metrics_result = cal_metrics(C)
    metrics_result = np.array(metrics_result)
    return metrics_result, acc



def print_history(result_array):
    fig = plt.figure()
    ax = plt.subplot(2,1,1)
    ax.plot(result_array[0])
    ax.plot(result_array[1])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    plt.legend(['Train_loss', 'Val_loss'], loc = 'upper right')
    ax = plt.subplot(2,1,2)
    ax.plot(result_array[2])
    ax.plot(result_array[3])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Weighted APUC', fontsize=12)
    fig.subplots_adjust(hspace=0.4)
    plt.legend(['Train_APUC', 'Val_APUC', 'Train_loss', 'Val_loss'], loc = 'upper right')

def py_auprc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    score = f1_score(y_true, y_pred, average='macro')
    score = score.astype(np.float32)
    return score
def tf_auprc(y_true, y_pred):
    return tf.numpy_function(py_auprc, (y_true, y_pred), tf.float32)

class F1S(tf.keras.metrics.Metric):
    def __init__(self, name="F1", **kwargs):
        super(F1S, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name="f1s", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.score.assign_add(tf_auprc(y_true, y_pred))
    def result(self):
        return self.score
    def reset_states(self):
        self.score.assign(0.0)

def train_model(model, x_train, y_train, x_val, y_val, x_test, y_test, save_path, EP, LR, BS, input_shape, output_dims, pic_path_ptb, pic_path_ptbxl, test = True):
    opt=Adam(lr=LR,decay=LR/EP)
    ME = F1S()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[ME])
    checkpoint = ModelCheckpoint(save_path, monitor='val_F1', verbose=1, save_best_only=True, save_weights_only=True,mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(x_train, y_train, epochs = EP, batch_size = BS, validation_data=(x_val, y_val), shuffle = True, callbacks = callbacks_list)
    history_dict = history.history
    ressult = np.array([history_dict['loss'], history_dict['val_loss'], history_dict['F1'], history_dict['val_F1']])
    np.save(save_path + 'history.npy', ressult)
    if test:
        best_model = build_model(input_shape, output_dims)
        best_model.load_weights(save_path)
        metrics_result_ptb, ptb_acc = evaluater(x_test[0], y_test[0], best_model, pic_path_ptb)
        metrics_result_ptbxl, ptbxl_acc = evaluater(x_test[1], y_test[1], best_model, pic_path_ptbxl)
        return history, metrics_result_ptb, ptb_acc, metrics_result_ptbxl, ptbxl_acc
        

if __name__ == '__main__':
    
    
    input_shape = (1200,3)
    output_dims = 4
    EP = 100
    LR = 1e-3
    BS = 512
    #leads idx = {'I':0, 'II':1, 'III':2, 'avR':3, 'avL':4, 'avF':5, 'V6':-1}
    used_list = np.array([0,1,2,3,4,5,-1])# all limb leads and V6
    save_path = 'path of saving model'
    model = build_model(input_shape, output_dims)
    
    # training data and validation data
    shaoxing = np.load("path of preprocessed data of Chapman-shaoxing") #(N, 12, 1200)
    shaoxing = shaoxing[:,used_list,:]
    val_rate = 0.2
    positive_list = [0,1,2,3]
    positive_rate_test = [0.5, 0.5/3, 0.5/3, 0.5/3]
    x_train, y_train, x_val, y_val = split_data(shaoxing, val_rate, positive_list, positive_rate_test)
    print(Counter(y_train))
    print(Counter(y_val))
    
    #ptbxl test
    ptbxl = np.load("path of preprocessed data of ptbxl") #(N, 12, 1200)
    ptbxl = ptbxl[:,used_list,:]
    val_rate = 1.0
    positive_list = [0,1,2,3]
    positive_rate_test = [0.5, 0.5/3, 0.5/3, 0.5/3]
    _, __, x_test_ptbxl, y_test_ptbxl = split_data(ptbxl, val_rate, positive_list, positive_rate_test)
    print(Counter(y_test_ptbxl))
    
    #ptb test
    ptb = np.load("path of preprocessed data of ptb") #(N, 12, 1200)
    ptb = ptb[:,used_list,:]
    val_rate = 1.0
    positive_list = [0,1,2,3]
    positive_rate_test = [0.5, 0.5/3, 0.5/3, 0.5/3]
    _, __, x_test_ptb, y_test_ptb = split_data(ptb, val_rate, positive_list, positive_rate_test)
    print(Counter(y_test_ptb))
    
    # III, avR and V6 as inout
    idx = [2,3,-1]
    x_train = np.transpose(x_train, [0,2,1])[:,:,np.array(idx)]
    x_val = np.transpose(x_val, [0,2,1])[:,:,np.array(idx)]
    x_test_ptbxl = np.transpose(x_test_ptbxl, [0,2,1])[:,:,np.array(idx)]
    x_test_ptb = np.transpose(x_test_ptb, [0,2,1])[:,:,np.array(idx)]
    
    
    y_train = to_categorical(y_train, num_classes = output_dims)
    y_val = to_categorical(y_val, num_classes = output_dims)
    x_test_list = [x_test_ptb, x_test_ptbxl]
    y_test_list = [y_test_ptb, y_test_ptbxl]
    pic_path_ptb = "path of ptb_fig"
    pic_path_ptbxl = "path of ptbxl_fig"
    history, result_ptb, acc_ptb, result_ptbxl, acc_ptbxl = train_model(model, x_train, y_train, x_val, y_val, x_test_list, y_test_list, save_path, EP, LR, BS, input_shape, output_dims, pic_path_ptb, pic_path_ptbxl)
    print(result_ptb)
    print(acc_ptb)
    
   