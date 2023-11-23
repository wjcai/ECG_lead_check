# -*- coding: utf-8 -*-

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from ILC_model import build_model
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.utils import to_categorical
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

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


def evaluater(x_test, y_test,model,path):
    y_pred = model.predict(x_test)
    num = y_pred.shape[-1]
    y_pred = np.argmax(y_pred, axis=1)
    acc = len(np.where(y_pred==y_test)[0])/y_pred.shape[0]
    C = confusion_matrix(y_test, y_pred, labels=range(num))
    #np.save("C:/Users/nyapass/Desktop/papers/comfuse/ptbxl_chest.npy", C)
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
    ticks = [0,1,2,3,4,5,6,7,8,9]
    plt.xticks(ticks, ticks) 
    plt.yticks(ticks, ticks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label' + '\n' + '\n' + name)
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
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = merge_label(y_pred, to_cat=False)
    y_test = merge_label(y_test, to_cat=False)
    C = confusion_matrix(y_test, y_pred, labels=range(2))

    plt.matshow(C, cmap=plt.cm.Reds) 

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    metrics_result = cal_metrics(C)
    metrics_result = np.array(metrics_result)
    return metrics_result



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

def train_model(model, x_train, y_train, x_val, y_val, x_test, y_test, save_path, EP, LR, BS, input_shape, output_dims, ptb_pic_path, ptbxl_pic_path):
    opt=Adam(lr=LR,decay=LR/EP)
    ME = F1S()

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[ME])
    checkpoint = ModelCheckpoint(save_path, monitor='val_F1', verbose=1, save_best_only=True, save_weights_only=True,mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(x_train, y_train, epochs = EP, batch_size = BS, validation_data=(x_val, y_val), shuffle = True, callbacks = callbacks_list)
    

    best_model = build_model(input_shape, output_dims)
    best_model.load_weights(save_path)
    metrics_result_ptb = evaluater(x_test[0], y_test[0], best_model, ptb_pic_path)
    metrics_result_ptbxl = evaluater(x_test[1], y_test[1], best_model, ptbxl_pic_path)
    return history, metrics_result_ptb, metrics_result_ptbxl





def rearrange_labels(labels):
    new_labels = []
    for i in range(labels.shape[0]):
        if labels[i]==0:
            new_labels.append(0)
        elif labels[i]==1:
            new_labels.append(1)
        elif labels[i]==2:
            new_labels.append(2)
        elif labels[i]==6:
            new_labels.append(3)
        elif labels[i]==7:
            new_labels.append(4)
        elif labels[i]==10:
            new_labels.append(5)
        elif labels[i]==11:
            new_labels.append(6)
        elif labels[i]==13:
            new_labels.append(7)
        elif labels[i]==14:
            new_labels.append(8)
        elif labels[i]==15:
            new_labels.append(9)
        else:
            print('error')
    new_labels = np.array(new_labels)
    return new_labels



if __name__ == '__main__':
    
   
    input_shape = (1200,6)
    output_dims = 10
    EP = 60
    LR = 1e-3
    BS = 128
    save_path = 'path of saving model'
    model = build_model(input_shape, output_dims)
    x_train = np.load('.../x_train.npy')
    y_train = np.load('.../y_train.npy')
    x_val = np.load('.../x_val.npy')
    y_val = np.load('.../y_val.npy')
    x_test_ptbxl = np.load(".../x_test.npy")
    y_test_ptbxl = np.load(".../y_test.npy")
    x_test_ptb = np.load(".../x_test.npy")
    y_test_ptb = np.load(".../y_test.npy")
    x_train = np.transpose(x_train, [0,2,1])
    x_val = np.transpose(x_val, [0,2,1])
    x_test_ptb = np.transpose(x_test_ptb, [0,2,1])
    x_test_ptbxl = np.transpose(x_test_ptbxl, [0,2,1])
    print(Counter(y_train))
    print(Counter(y_val))
    
    y_train = rearrange_labels(y_train)
    y_train = to_categorical(y_train, num_classes = output_dims)
    y_val = rearrange_labels(y_val)
    y_val = to_categorical(y_val, num_classes = output_dims)
    y_test_ptb = rearrange_labels(y_test_ptb)
    y_test_ptbxl = rearrange_labels(y_test_ptbxl)
    x_test_list = [x_test_ptb, x_test_ptbxl]
    y_test_list = [y_test_ptb, y_test_ptbxl]
    ptb_pic_path = '.../ptb.tiff'
    ptbxl_pic_path = '.../ptbxl.tiff'
    history, result_ptb, result_ptbxl = train_model(model, x_train, y_train, x_val, y_val, x_test_list, y_test_list, save_path, EP, LR, BS, input_shape, output_dims, ptb_pic_path, ptbxl_pic_path)