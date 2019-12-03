import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import random
import math
import time

from numpy.linalg import norm

def main():
    input_vector = np.array([random.uniform(-3.0, 3.0) for _ in range(16)])
    target_vector = np.zeros(len(input_vector))

    input_vector.sort()

    for i in range(len(target_vector)):
        target_vector[i] = 1 + math.sin(input_vector[i]*math.pi/6)

    hidden_layer_neurons = int(input("Enter number of neurons: "))
    if(hidden_layer_neurons not in (4, 8, 16)):
        print("ERROR! Wrong number of neurons!")
        exit()
    learning_rate = float(input("Enter learning rate: "))   
    
    weights_1 = np.array([random.uniform(-0.5, 0.5) for _ in range(hidden_layer_neurons)])
    bias_1 = np.array([random.uniform(-0.5, 0.5) for _ in range(hidden_layer_neurons)])
    
    weights_2 = np.array([random.uniform(-0.5, 0.5) for _ in range(hidden_layer_neurons)])
    bias_2 = random.uniform(-0.5, 0.5)
    
    epsilon = 1E-2
    
    iterations = 0

    while (True):
        index = 0
        successes = 0
        print("\n@ @ @ @ @ @ @ @ @ @ @ @ @")
        print("Iteration: ", iterations)
        for index in range(len(input_vector)):
            net_out = np.array([abs(input_vector[index] - weights_1[i])*bias_1[i] for i in range(hidden_layer_neurons)])
            
            hidden_output = np.exp(-1*np.power(net_out, 2))
            
            output_temp = np.dot(hidden_output, weights_2) + bias_2
            
            error = target_vector[index] - output_temp
            
            if (abs(error) <= epsilon):
                print("SUCCESS")
                successes+=1
                
            else:
                print("FAIL")
                sensitivity_2 = -2*1*error
                jacob_matrix = np.diag(np.array([-2*net_out[i]*hidden_output[i] for i in range(hidden_layer_neurons)]))
                sensitivity_1 = np.matmul(weights_2, jacob_matrix) * sensitivity_2
                
                weights_2 = weights_2 - learning_rate * sensitivity_2 * hidden_output
                bias_2 = bias_2 - learning_rate * sensitivity_2
                
                for i in range(hidden_layer_neurons):
                    weights_1[i] = weights_1[i] - learning_rate * sensitivity_1[i] * bias_1[i]*((weights_1[i] - input_vector[index])/abs(input_vector[index]-weights_1[i]))
                    bias_1[i] = bias_1[i] - learning_rate * sensitivity_1[i] * abs(input_vector[index] - weights_1[i])

            iterations+=1
        print("\n\n\n\tWINS: ", successes)
        time.sleep(0.4)
        if (successes >= len(input_vector)):
            break
    print("\n\nSUCCESSFULLY TRAINED NETWORK!")
    print("with a: ", learning_rate)
    print("with centers: ", hidden_layer_neurons)
    print("Final weights & biases:\n------\n")
    print("w1: ", weights_1, end='\n')
    print("b1: ", bias_1, end='\n')
    print("w2: ", weights_2, end='\n')
    print("b2: ", bias_2, end='\n')

    test_input = np.arange(-3, 3, 0.1)
    test_output = np.zeros(test_input.size)
    for index in range(len(test_output)):
        net_out = np.array([abs(test_input[index] - weights_1[i])*bias_1[i] for i in range(hidden_layer_neurons)])
            
        hidden_output = np.exp(-1*np.power(net_out, 2))
        
        test_output[index] = np.dot(hidden_output, weights_2) + bias_2

    
    sum_squared_error = 0
    for i in range(len(test_input)):
        sum_squared_error += pow(1+math.sin(test_input[i]*math.pi/6)-test_output[i], 2)
    print("Sum squared error: ", sum_squared_error)

    _, ax = plt.subplots()
    ax.plot(input_vector, target_vector, 'b^', label='g(p) function')
    ax.plot(test_input, test_output, 'r*', label='Neural Network response')
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    legend.get_frame().set_facecolor('C0')
    plt.grid()
    plt.axhline(linewidth='0.5', color='k')
    plt.axvline(linewidth='0.5', color='k')
    plt.show()
        
if __name__ == "__main__":
    main()