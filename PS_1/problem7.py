import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

#pylint: disable=no-member
def main():
    input_vector_p = np.arange(-3, 3, 0.2)
    weight_vector_sigmoid_layer = (-0.27, -0.41)
    weight_vector_lin_layer = (0.09, -0.17)
    bias_sigmoid_layer = (-0.48, -0.13)
    bias_lin_layer = 0.48

    net_val_sigmoid_layer = np.zeros((2, len(input_vector_p)))
    net_val_lin_layer = np.zeros(len(input_vector_p))

    output_sigmoid_layer = np.zeros((2, len(input_vector_p)))
    output_lin_layer = np.zeros(len(input_vector_p))

############## --INIT-- ##############
    for i in range(len(input_vector_p)):
        # Calculate net output (sigmoid)
        net_val_sigmoid_layer[0][i] = input_vector_p[i]*weight_vector_sigmoid_layer[0] + bias_sigmoid_layer[0]
        net_val_sigmoid_layer[1][i] = input_vector_p[i]*weight_vector_sigmoid_layer[1] + bias_sigmoid_layer[1]

        # Calculate sigmoid layer output
        output_sigmoid_layer[0][i] = sp.expit(net_val_sigmoid_layer[0][i])
        output_sigmoid_layer[1][i] = sp.expit(net_val_sigmoid_layer[1][i])

        #Calculate net output (linear)
        net_val_lin_layer[i] = output_sigmoid_layer[0][i]*weight_vector_lin_layer[0] + output_sigmoid_layer[1][i]*weight_vector_lin_layer[1] + bias_lin_layer

        #Calculate linear layer output
        output_lin_layer[i] = net_val_lin_layer[i]
    

    #Print sigm_net_1
    plt.figure('LOGSIG[0]')
    plt.plot(input_vector_p, net_val_sigmoid_layer[0], 'r', input_vector_p, output_sigmoid_layer[0], 'b')
    plt.xlabel('Input values')
    plt.ylabel('Logsig layer net output, value 1')
    plt.grid()
    plt.axhline(linewidth='0.5', color='r')
    plt.axvline(linewidth='0.5', color='r')


    #Print sigm_net_2
    plt.figure('LOGSIG[1]')
    plt.plot(input_vector_p, net_val_sigmoid_layer[1], 'r--', input_vector_p, output_sigmoid_layer[1], 'b--')
    plt.xlabel('Input values')
    plt.ylabel('Logsig layer net output, value 2')
    plt.grid()
    plt.axhline(linewidth='0.5', color='r')
    plt.axvline(linewidth='0.5', color='r')

    #Print lin_net
    plt.figure('PURELIN')
    plt.plot(input_vector_p, net_val_lin_layer, 'g', input_vector_p, output_lin_layer, 'k')
    plt.xlabel('Input values')
    plt.ylabel('Purelin layer net output')
    plt.axhline(linewidth='0.5', color='r')
    plt.axvline(linewidth='0.5', color='r')
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()