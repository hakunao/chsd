import os
import string
import numpy as np
from itertools import islice
import random
import csv
from time import time
import json
import pandas as pd 
import tensorflow as tf
import tensorflow.contrib.slim as slim
'''
    readcsv: Read feature tensors from csv data packet
    args:
        target: the directory that stores the csv files
        fealen: the length of feature tensor, related to to discarded DCT coefficients
    returns: (1) numpy array of feature tensors with shape: N x H x W x C
             (2) numpy array of labels with shape: N x 1 
'''
# def readcsv_(target, fealen=32):        #use the nummpy package
#     #read label
#     path  = target + '/label.csv'
#     label = np.genfromtxt(path, delimiter=',')
#     #read feature
#     feature = []
#     for dirname, dirnames, filenames in os.walk(target):
#         for i in range(0, len(filenames)-1):
#             if i==0:
#                 file = '/dc.csv'
#                 path = target + file
#                 featemp = np.genfromtxt(path, delimiter=',')
#                 feature.append(featemp)
#             else:
#                 file = '/ac'+str(i)+'.csv'
#                 path = target + file
#                 featemp = np.genfromtxt(path, delimiter=',')
#                 feature.append(featemp)          
#     return np.rollaxis(np.asarray(feature), 0, 3)[:,:,0:fealen], label

def swish_activation(x):
    return x * tf.nn.sigmoid(x)

def readcsv(target, fealen=32):         #use the pandas package     
    #read label
    path  = target + '/label.csv'
    label = np.genfromtxt(path, delimiter=',')
    #read feature
    feature = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in range(0, len(filenames)-1):
            if i==0:
                file = '/dc.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).values
                feature.append(featemp)
            else:
                file = '/ac'+str(i)+'.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).values
                feature.append(featemp)          
    return np.rollaxis(np.asarray(feature), 0, 3)[:,:,0:fealen], label
'''
    processlabel: adjust ground truth for biased learning
    args:
        label: numpy array contains labels
        cato : number of classes in the task
        delta1: bias for class 1
        delta2: bias for class 2
    return: softmax label with bias
'''
def processlabel(label, cato=2, delta1 = 0, delta2=0):
    softmaxlabel=np.zeros(len(label)*cato, dtype=np.float32).reshape(len(label), cato)
    for i in range(0, len(label)):
        if int(label[i])==0:
            softmaxlabel[i,0]=1-delta1
            softmaxlabel[i,1]=delta1
        if int(label[i])==1:
            softmaxlabel[i,0]=delta2
            softmaxlabel[i,1]=1-delta2
    return softmaxlabel

'''
    loss_to_bias: calculate the bias term for batch biased learning
    args:
        loss: the average loss of current batch with respect to the label without bias
        threshold: start biased learning when loss is below the threshold
    return: the bias value to calculate the gradient
'''
def loss_to_bias(loss,  alpha, threshold=0.3):
    if loss >= threshold:
        bias = 0
    else:
        bias = 1.0/(1+np.exp(alpha*loss))
    return bias


def residual_block(input, num_filters, layer_name):
    with tf.name_scope(layer_name) :
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0),
                            normalizer_fn=slim.batch_norm):

            input = slim.conv2d(input, num_filters, [1,1], scope=layer_name+'rise dimension')
            net = slim.conv2d(input, num_filters, [3, 3], scope=layer_name+'conv2_1')
            net = slim.conv2d(net, num_filters, [3, 3], activation_fn=None, scope=layer_name+'conv2_2')
            net += input

    return net



def squeeze_excitation_layer(input_x, input_dim,out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            with slim.arg_scope([slim.conv2d],  stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
                                        
                    net_16_i = slim.pool(input_x, [4, 4],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool4')
                    net_4 = slim.pool(input_x, [2, 2],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool5')
                    net_1 = slim.pool(input_x, [1, 1],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool6')

                    net_16 = slim.conv2d(net_16_i, 16*input_dim, [1,1], scope=layer_name+'rise_dimension_16')
                    net_4 = slim.conv2d(net_4, 4*input_dim, [1,1], scope=layer_name+'rise_dimension_4')
                    net_1 = slim.conv2d(net_1, 1*input_dim, [1,1], scope=layer_name+'rise_dimension_one')

                    squeeze = tf.concat([net_16, net_4, net_1], axis=3)
                    squeeze = tf.reduce_mean(squeeze, axis=[1, 2], keep_dims=True)
   
                    excitation = slim.fully_connected(squeeze, 6, activation_fn=tf.nn.relu, scope=layer_name+'_fully_connected3')
                    excitation = slim.fully_connected(squeeze, 4, activation_fn=tf.nn.relu, scope=layer_name+'_fully_connected4')
                    excitation = slim.fully_connected(squeeze, 3, activation_fn=tf.nn.relu, scope=layer_name+'_fully_connected5')
                    excitation = slim.fully_connected(squeeze, 1, activation_fn=tf.nn.relu, scope=layer_name+'_fully_connected')
                    adaptive_e = tf.reduce_mean(excitation) 

                    squeeze = tf.reshape(excitation, [-1, 1, 1, 1])

                    squeeze = slim.conv2d(squeeze, input_dim, [1,1], scope=layer_name+'rise_dimension_inital')
                    
                    # # Excitation
                    # excitation = slim.fully_connected(squeeze, out_dim // ratio, activation_fn=tf.nn.relu, scope=layer_name+'_fully_connected1')
                    # excitation = slim.fully_connected(excitation, out_dim, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected2')
                    
                    # Reshape
                    excitation = tf.reshape(squeeze, [-1, 1, 1, out_dim])
                    
                    # Scale
                    scale = input_x * excitation
                    # input_x = scale + net_16_i
                    adaptive_e = input_x * adaptive_e+ input_x
                    scale = scale + adaptive_e
                    return scale



def squeeze_excitation_layer_v1(input_x, layer_name, ratio=16, bias=1):
    with tf.name_scope(layer_name):
        c = 64

        net_16_a = slim.pool(input_x, [4, 4],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool4')
        net_4 = slim.pool(input_x, [2, 2],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool5')
        net_1 = slim.pool(input_x, [1, 1],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool6') 

        net_16 = slim.conv2d(net_16_a, 16*c, [1,1], scope=layer_name+'rise_dimension_16')
        net_4 = slim.conv2d(net_4, 4*c, [1,1], scope=layer_name+'rise_dimension_4')
        net_1 = slim.conv2d(net_1, 1*c, [1,1], scope=layer_name+'rise_dimension_one')

        squeeze_average = tf.concat([net_16, net_4, net_1], axis=3)

        net_16_m = slim.pool(input_x, [4, 4],  pooling_type='MAX',stride=1, padding='SAME', scope=layer_name+'pool1')
        net_4 = slim.pool(input_x, [2, 2],  pooling_type='MAX',stride=1, padding='SAME', scope=layer_name+'pool2')
        net_1 = slim.pool(input_x, [1, 1],  pooling_type='MAX',stride=1, padding='SAME', scope=layer_name+'pool3') 
        net_16_a += net_16_a + net_16_m
        
        net_16 = slim.conv2d(net_16_m, 16*c, [1,1], scope=layer_name+'rise_dimension_16_MAX')
        net_4 = slim.conv2d(net_4, 4*c, [1,1], scope=layer_name+'rise_dimension_4_MAX')
        net_1 = slim.conv2d(net_1, 1*c, [1,1], scope=layer_name+'rise_dimension_one_MAX')

        squeeze_max = tf.concat([net_16, net_4, net_1], axis=3)

        squeeze = squeeze_max + squeeze_average
        squeeze = tf.reduce_mean(squeeze, axis=[1, 2])

        excitation = slim.fully_connected(squeeze, 21*c // ratio, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected1')
        excitation = slim.fully_connected(excitation, c, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected2')

        adapative_e = slim.fully_connected(squeeze, 21*c // ratio, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected3')
        adapative_e = slim.fully_connected(adapative_e, c, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected5')
        adapative_e = slim.fully_connected(adapative_e, 1, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected4')

        excitation = tf.reshape(excitation,[-1,1,1,c])
        # excitation = tf.reshape()
        adapative_e = tf.reduce_mean(adapative_e)
        
        # Scale input_x with calculated weights
        scale = net_16_a * excitation + net_16_a
        scale = scale * adapative_e 
        return scale

def squeeze_excitation_layer_v2(input_x, layer_name,ratio=16, bias=1):
    with tf.name_scope(layer_name):
        c = 128

        net_16_a = slim.pool(input_x, [4, 4],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool4')
        net_4 = slim.pool(input_x, [2, 2],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool5')
        net_1 = slim.pool(input_x, [1, 1],  pooling_type='AVG',stride=1, padding='SAME', scope=layer_name+'pool6') 

        net_16 = slim.conv2d(net_16_a, 16*c, [1,1], scope=layer_name+'rise_dimension_16')
        net_4 = slim.conv2d(net_4, 4*c, [1,1], scope=layer_name+'rise_dimension_4')
        net_1 = slim.conv2d(net_1, 1*c, [1,1], scope=layer_name+'rise_dimension_one')

        squeeze_average = tf.concat([net_16, net_4, net_1], axis=3)

        net_16_m = slim.pool(input_x, [4, 4],  pooling_type='MAX',stride=1, padding='SAME', scope=layer_name+'pool1')
        net_4 = slim.pool(input_x, [2, 2],  pooling_type='MAX',stride=1, padding='SAME', scope=layer_name+'pool2')
        net_1 = slim.pool(input_x, [1, 1],  pooling_type='MAX',stride=1, padding='SAME', scope=layer_name+'pool3') 
        net_16_a += net_16_a + net_16_m
        
        net_16 = slim.conv2d(net_16_m, 16*c, [1,1], scope=layer_name+'rise_dimension_16_MAX')
        net_4 = slim.conv2d(net_4, 4*c, [1,1], scope=layer_name+'rise_dimension_4_MAX')
        net_1 = slim.conv2d(net_1, 1*c, [1,1], scope=layer_name+'rise_dimension_one_MAX')


        squeeze_max = tf.concat([net_16, net_4, net_1], axis=3)

        squeeze = squeeze_max + squeeze_average
        squeeze = tf.reduce_mean(squeeze, axis=[1, 2])

        excitation = slim.fully_connected(squeeze, 21*c // ratio, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected1')
        excitation = slim.fully_connected(excitation, c, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected2')
        
        excitation = tf.reshape(excitation,[-1,1,1,c])

        adapative_e = slim.fully_connected(squeeze, 21*c // ratio, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected3')
        adapative_e = slim.fully_connected(adapative_e, c, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected5')
        adapative_e = slim.fully_connected(adapative_e, 1, activation_fn=tf.nn.sigmoid, scope=layer_name+'_fully_connected4')

        adapative_e = tf.reduce_mean(adapative_e)
        
        # Scale input_x with calculated weights
        scale = net_16_a * excitation + net_16_a
        scale = scale * adapative_e 
        return scale




'''
    forward: define the neural network architecute
    args:
        input: feature tensor batch with size B x H x W x C
        is_training: whether the forward process is training, affect dropout layer
        reuse: undetermined
        scope: undetermined
    return: prediction socre(s) of input batch
'''
def forward(input, is_training=True, reuse=False, scope='model', flip=True):
    if flip == True:
        input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
        input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            
            # 12
            net = slim.conv2d(input, 32, [3, 3], scope='conv1_1_i')
            net = slim.conv2d(input, 32, [3, 3], scope='conv1_2_i')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')

            input = slim.conv2d(net, 64, [1,1], scope='rise_dimension')
            net = slim.conv2d(input, 64, [3, 3], scope='conv2_1')
            se = squeeze_excitation_layer_v1(net,'eca_2')
            net = input + se
            
            input = slim.conv2d(net, 64, [1,1], scope='rise_dimension_2')
            net = slim.conv2d(input, 64, [3, 3], scope='conv3_1')
            se = squeeze_excitation_layer_v1(net,'eca_3')
            net = input + se
           
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool3')
            input = slim.conv2d(net, 128, [1,1], scope='rise_dimension_3')
            net = slim.conv2d(input, 128, [3, 3], scope='conv4_1')
            se = squeeze_excitation_layer_v2(net,'eca_4')
            net = input + se

            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool3')
            net = slim.conv2d(input, 256, [3, 3], scope='conv5_1')

            net = tf.reduce_mean(net, axis=[1, 2], keep_dims=True)
            net = slim.flatten(net)

            w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='fc1')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')

            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc4')
    return predict




'''
    data: a class to handle the training and testing data, implement minibatch fetch
    args: 
        fea: feature tensor of whole data set
        lab: labels of whole data set
        ptr: a pointer for the current location of minibatch
        maxlen: length of entire dataset
        preload: in current version, to reduce the indexing overhead of SGD, we load all the data into memeory at initialization.
    methods:
        nextinstance():  returns a single instance and its label from the training set, used for SGD
        nextbatch(): returns a batch of instances and their labels from the training set, used for MGD
            args: 
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label
        sgd_batch(): returns a batch of instances and their labels from the trainin set randomly, number of hs and nhs are equal.
            args:
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label

'''
class data:
    def __init__(self, fea, lab, preload=False):
        self.ptr_n=0
        self.ptr_h=0
        self.ptr=0
        self.dat=fea
        self.label=lab
        with open(lab) as f:
            self.maxlen=sum(1 for _ in f)                       #get the max length of the label
        if preload:
            print("loading data into the main memory...")
            self.ft_buffer, self.label_buffer=readcsv(self.dat) #get the data and label,the data is N*H*W*C


    # 闂備緡鍋勯ˇ鎵偓姘ュ妽缁嬪鍩€椤掍胶鈻旀い蹇撳暟娴兼劙鎮楀☉铏 
    def nextinstance(self):
        temp_fea=[]
        label=None
        # 闂傚倸鎳庣换鎴濃攦閳ь剟姊洪銏╂Ч閻庢哎鍔嶇粙澶愬焵椤掍胶鈻旀い蹇撴噽閸ㄨ偐绱掑☉娆忓姢妞ゆ洦鍠楅幆鏃堝籍閸屾稓鏆犵紓浣风┒閸ㄥ湱妲�?
        idx=random.randint(0,self.maxlen)
        for dirname, dirnames, filenames in os.walk(self.dat):
            for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))        
        with open(self.label) as l:
            temp_label=np.asarray(list(l)[idx]).astype(int)
            if temp_label==0:
                label=[1,0]
            else:
                label=[0,1]
        return np.rollaxis(np.array(temp_fea),0,3),np.array([label])

# 闂佸湱鐟抽崱鈺傛杸闂傚倸鎳庣换鎴濃攦閳ь剚鎱ㄦ繝搴＄仩閻庣娅曠粙澶屸偓锝庡枟椤庯拷
    def sgd(self, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
        length=labelist.size
        idx=random.randint(0, length-1)
        temp_label=labelist[idx]
        if temp_label==0:
            label=[1,0]
        else:
            label=[0,1]
        ft= self.ft_buffer[idx]

        return ft, np.array(label)
    def sgd_batch_2(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        label = processlabel(label,2, 0,0 )
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label


    def sgd_batch(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        #label = processlabel(label,2, delta1, delta2)
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_beta: returns the balalced batch, used for training only
    '''
    def nextbatch_beta(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size

        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch//2
            if num>=n_length or num>=h_length:
                print('ERROR:Batch size exceeds data size')
                print('Abort.')
                quit()
            else:
                if self.ptr_n+num <n_length:
                    idxn = labexn[self.ptr_n:self.ptr_n+num]
                elif self.ptr_n+num >=n_length:
                    idxn = np.concatenate((labexn[self.ptr_n:n_length], labexn[0:self.ptr_n+num-n_length]))
                self.ptr_n = update_ptr(self.ptr_n, num, n_length)
                if self.ptr_h+num <h_length:
                    idxh = labexh[self.ptr_h:self.ptr_h+num]
                elif self.ptr_h+num >=h_length:
                    idxh = np.concatenate((labexh[self.ptr_h:h_length], labexh[0:self.ptr_h+num-h_length]))
                self.ptr_h = update_ptr(self.ptr_h, num, h_length)
                #print self.ptr_n, self.ptr_h
                label = np.concatenate((np.zeros(num), np.ones(num)))
                #label = processlabel(label,2, delta1, delta2)
                ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
                ft_batch_nhs = self.ft_buffer[idxn]
                label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_without_balance: returns the normal batch. Suggest to use for training and validation
    '''
    def nextbatch_without_balance_alpha(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        if self.ptr + batch < self.maxlen:
            label = self.label_buffer[self.ptr:self.ptr+batch]
            ft_batch = self.ft_buffer[self.ptr:self.ptr+batch]
        else:
            label = np.concatenate((self.label_buffer[self.ptr:self.maxlen], self.label_buffer[0:self.ptr+batch-self.maxlen]))
            ft_batch = np.concatenate((self.ft_buffer[self.ptr:self.maxlen], self.ft_buffer[0:self.ptr+batch-self.maxlen]))
        self.ptr = update_ptr(self.ptr, batch, self.maxlen)
        return ft_batch, label
    def nextbatch(self, batch, channel=None, delta1=0, delta2=0):
        #print('recommed to use nextbatch_beta() instead')
        databat=None
        temp_fea=[]
        label=None
        if batch>self.maxlen:
            print('ERROR:Batch size exceeds data size')
            print('Abort.')
            quit()
        if self.ptr+batch < self.maxlen:
            #processing labels
            with open(self.label) as l:
                temp_label=np.asarray(list(l)[self.ptr:self.ptr+batch])
                label=processlabel(temp_label, 2, delta1, delta2)
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
            self.ptr=self.ptr+batch
        elif (self.ptr+batch) >= self.maxlen:
            
            #processing labels
            with open(self.label) as l:
                a=np.genfromtxt(islice(l, self.ptr, self.maxlen),delimiter=',')
            with open(self.label) as l:
                b=np.genfromtxt(islice(l, 0, self.ptr+batch-self.maxlen),delimiter=',')
            #processing data
            if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                temp_label=b
            elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                temp_label=a
            else:
                temp_label=np.concatenate((a,b))
            label=processlabel(temp_label,2, delta1, delta2)
            #print label.shape
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, None, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, 0, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
            self.ptr=self.ptr+batch-self.maxlen
        #print np.asarray(temp_fea).shape
        return np.rollaxis(np.asarray(temp_fea), 0, 3)[:,:,0:channel], label



# def forward(input, is_training=True, reuse=False, scope='model', flip=False):
#     if flip == True:
#         input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
#         input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)

#     with tf.variable_scope(scope, reuse=reuse):
#         with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu, stride=1, padding='SAME',
#                             weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                             biases_initializer=tf.constant_initializer(0.0)):
            
#             # 12
#             net = slim.conv2d(input, 32, [3, 3], scope='conv1_1')
#             net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')

#             input = slim.conv2d(net, 64, [1,1], scope='rise_dimension')
#             net = slim.conv2d(input, 64, [3, 3], scope='conv2_1')
#             se = squeeze_excitation_layer(input, 64, 16, 'se1')
#             net += se
#             net += input

            
#             net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')

#             # 6
#             input = slim.conv2d(net, 128, [1,1], scope='rise_dimension_2')
#             net = slim.conv2d(input, 128, [3, 3], scope='conv3_1')
#             se = squeeze_excitation_layer(input, 128, 16, 'se2')
#             net += se
#             net += input
            

#             net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool3')
#             net = slim.conv2d(input, 256, [3, 3], scope='conv4_1')
            
#             net = tf.reduce_mean(net, axis=[1, 2], keep_dims=True)
#             net = slim.flatten(net)

#             w_init = tf.contrib.layers.xavier_initializer(uniform=False)
#             net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='fc1')
#             net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')


#             predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
#     return predict