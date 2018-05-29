
# coding: utf-8

# In[9]:


'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    
    sig = 1.0 / (1.0 + np.exp(np.multiply(-1.0,z)))
    return  sig
    
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    obj_val = 0
    
    targetbias = np.ones(training_data.shape[0])
    
    new_traindata = np.column_stack([training_data,targetbias]) 
          
    linear_inputdata = np.dot(new_traindata,w1.T)
    sig = sigmoid(linear_inputdata)
    output_activation = np.column_stack([sig,targetbias])
    output_layer = sigmoid(np.dot(output_activation,w2.T))
        
    target_value = np.zeros((len(new_traindata),2))
    for i in range (0,len(new_traindata)):
        target_value[i][int(training_label[i])]=1
               
    #Error    

    function_error = np.sum(np.multiply(target_value,np.log(output_layer)) + np.multiply((1-target_value),np.log(1-output_layer)))
    error = np.negative(function_error/(len(new_traindata)))

    #Gradiant Descent 
    
    delta = output_layer - target_value
    derivative_wrt_w2 = np.dot(delta.T,output_activation)
    derivative_errorfunction = ((np.dot(delta,w2))*((1 - output_activation)*output_activation))
    derivative_wrt_w1 = np.dot(derivative_errorfunction.T,new_traindata)
    derivative_wrt_w1 = derivative_wrt_w1[0:n_hidden,:]
        
    #Regularization
         
    regularization_term = (lambdaval/(2*len(new_traindata)))*(np.sum(np.square(w1))+np.sum(np.square(w2)))
    obj_val = error + regularization_term
    
    grad_w1 = (derivative_wrt_w1 + (lambdaval*w1))/len(new_traindata)
    grad_w2 = (derivative_wrt_w2 + (lambdaval*w2))/len(new_traindata)
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)

    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    
    target_val = np.ones((data.shape[0], 1),dtype = np.uint8)
    complete_data = np.column_stack([data, target_val])
    sig_1 = np.dot(complete_data, w1.transpose())
    sig_output = sigmoid(sig_1)      
    
    target_bias = np.ones((sig_output.shape[0], 1),dtype = np.uint8)   
    output = np.column_stack([sig_output, target_bias])
    sig_2 = np.dot(output, w2.transpose())

    result = sigmoid(sig_2)       
    labels = np.argmax(result, axis=1)  

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

