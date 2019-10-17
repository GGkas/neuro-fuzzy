import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import statistics as stat


def adaline_training(weight, bias, target_vector, p_vector, theory_max_lr):
    epoch = 1
    faults = np.zeros(40)
    falses = 1
    errors = np.zeros((40, 2))
    error = 0
    alpha = theory_max_lr/2
    epsilon = 1E-3

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
                print(weight)
                print(bias)
        
        print('\n\nEpoch ' + str(epoch) + ' finished, with ' + str(faults[index]) + ' faults.')
        index+=1
        epoch+=1

    return weight, bias

def main():
    target_vector = np.array([1,-1])
    p1 = np.array([1,1])
    p2 = np.array([1,-1])
    p_vector = np.array([p1,p2])
    
    
    """
    step1: using sympy to find the MSE and maximum a
    """
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

    ######### QUESTION B, C ###########
    weight_B, bias_B = adaline_training(np.array([0, 0]), 0, target_vector, p_vector, max_learning_rate)
    weight_C, bias_C = adaline_training(np.array([1, 1]), 1, target_vector, p_vector, max_learning_rate)

    print('Final weight for B: ' + str(weight_B))
    print('Final bias for B: ' + str(bias_B))
    print('Final weight for C: ' + str(weight_C))
    print('Final bias for C: ' + str(bias_C))

    plt.figure("Final Decision Boundary for B")
    x = np.arange(-3, 3, 0.1)
    plt.plot(x, -(weight_B[0]/weight_B[1])*x - bias_B/weight_B[1], 'k', p1[0], p1[1], 'r*', p2[0], p2[1], 'b*')
    plt.grid()
    
    plt.figure("Final Decision Boundary for C")
    plt.plot(x, -(weight_C[0]/weight_C[1])*x - bias_C/weight_C[1], 'k', p1[0], p1[1], 'r*', p2[0], p2[1], 'b*')
    plt.grid()
    
    plt.axhline(linewidth='0.5', color='r')
    plt.axvline(linewidth='0.5', color='r')
    plt.show()
    
if __name__ == "__main__" :
    main()