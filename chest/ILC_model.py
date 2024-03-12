# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:35:29 2023

@author: nyapass
"""
import tensorflow as tf

# compute the correlation coefficient between two feature maps
def compute_cof(x,y):
    '''
    x,y: two feature map with the same shape, tensor
    '''
    x = tf.keras.layers.Flatten()(x)
    y = tf.keras.layers.Flatten()(y)
    fz = tf.reduce_sum(x*y, axis=-1)
    fm = tf.norm(x, axis=-1)*tf.norm(y, axis=-1)
    return fz/fm

def Conv_1D_Block(x, kernel_num, kernel_size, strides):
    '''
    N: sample number
    L: feature length
    C: channel number
    x: input tensor, (N, L, C)
    model_width: filter number of convolution layer, int
    kernel: kernel length of convolution layer, int
    strides: stride of convolution layer, int
    '''
    x = tf.keras.layers.Conv1D(kernel_num, kernel_size, strides=strides, padding="same")(x) #kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Activation('swish')(x)
    return x

# define correlation coefficient layer
def cof_layer(fms):
    '''
    fms: feature map list
    '''
    cof_list = []
    for i in range(len(fms)-1):
        #cof_sample = []
        for j in range(i+1,len(fms)):
            cof = compute_cof(fms[i], fms[j])
            cof = tf.expand_dims(cof, axis=-1)
            cof_list.append(cof)
    return cof_list


def stem(inputs, num_filters, filter_len):
    '''
    inputs: input tensor, (sample number, signal length, channel number)
    num_filters: filter number of convolution layer, int
    filter_len: kernel length of convolution layer, int
    '''
    conv = Conv_1D_Block(inputs, num_filters, filter_len, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="same")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    return pool

def conv_block(x, num_filters, bottleneck=True):
    '''
    N: sample number
    L: feature length
    C: channel number
    x: input tensor, (N, L, C)
    num_filters: filter number of convolution layer, int
    bottleneck: whether use bottleneck filter, bool
    '''
    if bottleneck:
        num_filters_bottleneck = num_filters * 4
        x = Conv_1D_Block(x, num_filters_bottleneck, 1, 1)

    out = Conv_1D_Block(x, num_filters, 7, 1)
    return out


def dense_block(x, num_filters, num_layers, bottleneck=True):
    '''
    N: sample number
    L: feature length
    C: channel number
    x: input tensor, (N, L, C)
    num_filters: filter number of convolution layer, int
    num_layers: number of convolution layer, int
    bottleneck: whether use bottleneck filter, bool
    '''
    cb_list = []
    for i in range(num_layers):
        cb = conv_block(x, num_filters, bottleneck=bottleneck)
        cb_list.append(cb)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)
    return x, cb_list


 
# define branch architecture
def branch_model(inputs_shape):
    inputs = tf.keras.Input(inputs_shape)
    stem_block = stem(inputs, num_filters=16, filter_len=11)
    dense_op, conv_op_list = dense_block(stem_block, num_filters = 8, num_layers = 3, bottleneck=True)
    conv_op_list.append(stem_block)#在倒数第二个
    conv_op_list.append(dense_op)#在最后
    model = tf.keras.Model(inputs, conv_op_list)
    return model

# build ILC model
def build_model(input_shape, output_dims):
    '''
    input_shape: input shape of model, tuple with shape of (length, channel)
    output_dims: number of the output categories, int
    '''
    inputs = tf.keras.Input(input_shape)
    input_list = tf.unstack(inputs, axis=-1)
    input_list_exp = [tf.expand_dims(i, -1) for i in input_list]
    stem_ops = []
    conv1_ops = []
    conv2_ops = []
    conv3_ops = []
    dense_ops = []
    branch = branch_model((input_list_exp[0].shape[1], input_list_exp[0].shape[-1]))
    for i in range(len(input_list_exp)):
        conv1_op, conv2_op, conv3_op, stem_op, dense_op = branch(input_list_exp[i])
        stem_ops.append(stem_op)
        conv1_ops.append(conv1_op)
        conv2_ops.append(conv2_op)
        conv3_ops.append(conv3_op)
        dense_ops.append(dense_op)

    cof_list1 = cof_layer(conv1_ops)
    cof_list2 = cof_layer(conv2_ops)
    cof_list3 = cof_layer(conv3_ops)

    cof_alllist = cof_list1 + cof_list2 + cof_list3

    cofs = tf.keras.layers.Concatenate(axis=-1)(cof_alllist)
    fms = tf.keras.layers.Concatenate(axis=-1)(dense_ops)
    fms = tf.keras.layers.GlobalAveragePooling1D()(fms)
    all_feature = tf.keras.layers.Concatenate(axis=-1)([fms, cofs])
    all_feature = tf.keras.layers.Dense(64, activation='relu')(all_feature)
    all_feature = tf.keras.layers.Dense(output_dims, activation= 'softmax')(all_feature)
    model = tf.keras.Model(inputs, all_feature)

    return model
    




if __name__ == '__main__':
    
    input_shape = (1200,6)
    output_dims = 10
    model = build_model(input_shape, output_dims)
    model.summary()
    