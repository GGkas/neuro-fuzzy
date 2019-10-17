#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import statistics as stat
import random
import warnings as warn

def adaline_training(weight, bias, target_vector, p_vector, theory_max_lr):
    epoch = 1
    faults = np.zeros(40)
    falses = 1
    errors = np.zeros((40, 2))
    error = 0
    alpha = theory_max_lr/2
    epsilon = 1E-3
    weight_vector = np.zeros((40,2))
    weight_vector[0] = weight
    index = 0
    while(falses > 0 and index < 40):
        falses = 0
        for i in range(len(p_vector)):
            a = np.dot(p_vector[i], weight) + bias
            error = target_vector[i] - a
            if (abs(error) > epsilon):
                errors[index][i] = error
                faults[index]+=1
                falses+=1
                bias += 2*alpha*error
                weight = weight + 2*alpha*error*p_vector[i]
                weight_vector[index+1]= weight
                print(weight)
                print(bias)
        
        print('\n\nEpoch ' + str(epoch) + ' finished, with ' + str(faults[index]) + ' faults.')
        index+=1
        epoch+=1

    weight_vector_del = np.delete(weight_vector, np.s_[index:], axis=0)

    return weight,bias,weight_vector_del



def main():
    
    target_vector = np.array([-1,1])
    p1 = np.array([1,2])
    p2 = np.array([-2,1])
    p_vector = np.array([p1,p2])

    w1, w2 = sym.symbols('w1 w2')
    
    sym_weight = [w1, w2]
    C = stat.mean(target_vector**2)

    TZ_ARRAY = np.array([target_vector[0]*p_vector[0], target_vector[1]*p_vector[1]])
    row_sum_of_array = np.zeros(len(TZ_ARRAY))
    for row in TZ_ARRAY:
        row_sum_of_array += row
    D = -2*(np.array(row_sum_of_array/TZ_ARRAY.shape[0], dtype=np.int32))
    print(D)
    print(C)
    R = np.array((np.array(p1.reshape(-1, 1))@np.array([p1]) + np.array(p2.reshape(-1, 1))@np.array([p2]))/p_vector.shape[0], dtype=np.int32)
    print(R)
    A = 2*R
    print(A)
    
    F = C + np.dot(D, sym_weight) + np.dot(np.dot(np.array([sym_weight]), R), np.array(sym_weight).reshape(-1, 1))
    print('F(w1, w2) = ' + str(F).strip('[]'))

    eigenvals, eigenvecs = np.linalg.eig(A)
    del eigenvecs
    max_learning_rate = 1/(min(eigenvals))
    print(max_learning_rate)

    func = sym.lambdify((w1, w2), 2*w1**2 + 2*w2**2 + 2*w1 + 1, 'numpy')
    xaxis = np.linspace(-20, 20, 1000)
    yaxis = np.linspace(-16, 16, 1000)
    x, y = np.meshgrid(xaxis, yaxis)
    plt.contour(x, y, func(x, y))
    plt.grid()
    plt.axhline(linewidth='0.5', color='r')
    plt.axvline(linewidth='0.5', color='r')
    plt.show()
    
    #################### QUESTION B #######################
    W = np.array([random.uniform(0, 1),random.uniform(0, 1)])
    B = random.uniform(0, 1)
    
    weight, bias, weight_vector = adaline_training(W, B, target_vector, p_vector, max_learning_rate)
    
    x = np.arange(-3, 3, 0.2)
    plt.figure()
    plt.plot(x, -(weight[0]/weight[1])*x - bias/weight[1], 'k', p1[0], p1[1], 'r*', p2[0], p2[1], 'b*')
    plt.title('Decision boundary for initial weight of (1 0) and initial bias of 0.5')
    plt.grid()
    plt.axhline(linewidth='0.5', color='r')
    plt.axvline(linewidth='0.5', color='r')
    plt.show()
  
    print('Final weight: ' + str(weight))
    print('Final bias: ' + str(bias))
    
    #####################  QUESTION C ######################
    
    W = np.array([0,1])
    alpha = 0.1
    B = 0
    
    weight, bias, weight_vector  = adaline_training(W, B, target_vector, p_vector, alpha)
    

    xaxis = np.linspace(-3, 3, 1000)
    yaxis = np.linspace(-2, 2, 1000)
    x, y = np.meshgrid(xaxis, yaxis)
    plt.contour(x, y, func(x, y))
    plt.margins(0.25, -0.25)
    with warn.catch_warnings():
        warn.simplefilter("ignore")
        plt.plot(weight_vector[:, 0], weight_vector[:, 1], 'g^', weight_vector[:, 0], weight_vector[:, 1], 'k', W[0], W[1], 'b*', weight[0], weight[1], 'r*', linestyle='default', lw=1.0)
    plt.grid()
    plt.axhline(linewidth='0.5', color='r')
    plt.axvline(linewidth='0.5', color='r')
    plt.show()
    
if __name__ == "__main__" :
    main()