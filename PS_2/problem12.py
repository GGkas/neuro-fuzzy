import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy.linalg import norm, pinv

def main():
    pass
    weight = np.array([0, -2]) # weight[0]->1st layer, weight[1] 2nd
    bias = np.array([0, 1])
    inputs = np.array([-1, 1])
    targets = np.array([0, 1])

    epsilon = 1E-3
    alpha = 1
    
    for i in range(0, 2):
        # Forward pass
        net_1 = abs(inputs[i] - weight[0])*bias[0]
        print(str(net_1) + '\n\n')
        hidd_out = np.exp(-net_1**2)
        print(str(hidd_out) + '\n\n')

        output = hidd_out*weight[1] + bias[1]

        error = targets[i] - output
        if (abs(error) < epsilon):
            print("Success with epsilon! Final weight and bias vectors are: " + str(weight) + ", " + str(bias))
            break
        else:
            # Backward pass
            sensitivity_2 = -2*error
            sensitivity_1 = -2*net_1*hidd_out*sensitivity_2*weight[1]

            weight[1] -= alpha*sensitivity_2*hidd_out
            bias[1] -= alpha*sensitivity_2

            weight[0] -= alpha*sensitivity_1*inputs[i]
            bias[0] -= alpha*sensitivity_1
    print("Final weights and biases after 2 iterations are: " + str(weight) + ", " + str(bias))

if __name__ == "__main__":
    main()