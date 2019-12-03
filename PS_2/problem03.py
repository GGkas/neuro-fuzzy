#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:54:07 2019

@author: tsiv
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.axes
import random
import math
import time

#pylint: disable=no-member
def training_function(input_vec, target, hidden_S, learning_rate):
    if (hidden_S < 2 and hidden_S > 10):
        print ("Error: wrong number of neurons!")
        return None, None, None, None, None
    weights_1 = np.zeros((1, hidden_S))
    weights_2 = np.zeros((1, hidden_S))
    bias_1 = np.zeros((1, hidden_S))
    bias_2 = 0

    for i in range(hidden_S):
        weights_1[0][i] = random.uniform(-0.5, 0.5)
        weights_2[0][i] = random.uniform(-0.5, 0.5)
        bias_1[0][i] = random.uniform(-0.5, 0.5)
    bias_2 = random.uniform(-0.5, 0.5)

    epsilon = 1E-6
    index = 0
    output_list = np.zeros(len(input_vec))
    
    while(index < len(input_vec)):
        epoch = 1
        while(1):
            net_out = np.add(input_vec[index]*np.copy(weights_1), bias_1)
    
            hidden_output = np.zeros((1, hidden_S))
            for i in range(hidden_S):
                hidden_output[0][i] = sp.expit(net_out[0][i])
        
            temp_out = np.dot(weights_2, hidden_output.transpose()) + bias_2
    
            error = target[index] - temp_out
    
            if (abs(error) <= epsilon):
                print('Good job!')
                break
            else:
                print("epoch: ",epoch)
                sensitivity_2 = -2*1*error
                jacob_matrix = np.diag(np.array([(1-S)*S for S in hidden_output[0]]))
                sensitivity_1 = np.matmul(jacob_matrix, weights_2.transpose())*sensitivity_2
    
                weights_2 = np.add(weights_2, -learning_rate * sensitivity_2 * hidden_output[0])
                bias_2 = bias_2 - learning_rate * sensitivity_2
    
                weights_1 = np.add(weights_1, -learning_rate*input_vec[index]*sensitivity_1.transpose())
                bias_1 = np.add(bias_1, -learning_rate*sensitivity_1.transpose())
                epoch += 1

        net_out = np.add(input_vec[index]*np.copy(weights_1), bias_1)

        hidden_output = np.zeros((1, hidden_S))
        for i in range(hidden_S):
            hidden_output[0][i] = sp.expit(net_out[0][i])

        output_list[index] = np.dot(weights_2, hidden_output.transpose()) + bias_2
        index += 1
    return output_list

def main():
    input_vector = np.arange(-2, 2, 0.1)
    target_vector = np.zeros(len(input_vector))

    for i in range(len(target_vector)):
        target_vector[i] = 1 + math.sin(input_vector[i]*math.pi/2)

    hidden_layer_neurons = int(input("Enter number of neurons: "))
    learning_rate = float(input("Enter learning rate: "))
    output_list = training_function(input_vector, target_vector, hidden_layer_neurons, learning_rate)

    ## PLOTTING
    _, ax = plt.subplots()
    ax.plot(input_vector, target_vector, 'b^', label='g(p) function')
    ax.plot(input_vector, output_list, 'r*', label='Neural Network response')
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    legend.get_frame().set_facecolor('C0')
    plt.grid()
    plt.axhline(linewidth='0.5', color='k')
    plt.axvline(linewidth='0.5', color='k')
    plt.show()

if __name__ == "__main__":
    main()