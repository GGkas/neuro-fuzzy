import numpy as np
import random
from numpy.linalg import eig

def main():
    H_mtx = np.array([[3, 1], [1, 3]])

    eigenvalues, _ = eig(H_mtx)
    print(str(eigenvalues))

    alpha = 1
    momentum_coeff = [0, 0.6] # 2 possible values

    gamma_stable = np.max(np.array([((-1+alpha*eigenval)/(1+alpha*eigenval))**2 for eigenval in eigenvalues]))
    print(gamma_stable)
    
    for i in range(len(momentum_coeff)):
        print("For momentum = " + str(momentum_coeff[i]) + ", ", endl=' ')
        if (momentum_coeff[i] < gamma_stable):
            print("the algorithm is stable!")
        else:
            print("the algorithm is NOT stable!")
if __name__ == "__main__":
    main()