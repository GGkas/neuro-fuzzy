import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

def main ():
    lambdavar_1, alpha_1 = (3, 0.3)
    lambdavar_2, alpha_2 = (lambdavar_1, 0.9)
    lambdavar_3, alpha_3 = (3.5, alpha_2)
    
    print('Running test A')

    fin_array_1 = np.zeros( 150 )
    fin_array_2 = np.zeros( 150 )
    fin_array_3 = np.zeros( 150 )

    start_value = 0.5

    fin_array_1[0] = start_value
    for k in range(149):
        temp = lambdavar_1*(fin_array_1[k]**alpha_1)*(1 - fin_array_1[k]**alpha_1)
        fin_array_1[k+1] = temp 

    fin_array_2[0] = start_value
    for k in range(149):
        temp = lambdavar_2*(fin_array_2[k]**alpha_2)*(1 - fin_array_2[k]**alpha_2)
        fin_array_2[k+1] = temp 
    

    fin_array_3[0] = start_value
    for k in range(149):
        temp = lambdavar_3*(fin_array_3[k]**alpha_3)*(1 - fin_array_3[k]**alpha_3)
        fin_array_3[k+1] = temp 

    plt.figure()
    plt.plot(np.arange(0.5, 1, 0.5/149), fin_array_1[1:], 'r*', np.arange(0.5, 1, 0.5/149), fin_array_1[1:], 'r', np.arange(0.5, 1, 0.5/149), fin_array_2[1:], 'g^', np.arange(0.5, 1, 0.5/149), fin_array_2[1:], 'g', np.arange(0.5, 1, 0.5/149), fin_array_3[1:], 'bo', np.arange(0.5, 1, 0.5/149), fin_array_3[1:], 'k', linestyle='default', lw=1.0)
    plt.grid()
    plt.title('Output trajectories for pairs of (λ, α) = (3, 0.3), (3, 0.9), (3.5, 0.9) respectively')
    plt.show()

    print('Running test B')

    lambda_test1, alpha_test1 = (3.6, 0.73)
    lambda_test2, alpha_test2 = (3.7, 0.73)

    test_array1 = np.zeros( 150 )
    test_array2 = np.zeros( 150 )

    test_array1[0] = test_array2[0] = start_value

    for k in range(149):
        test_array1[k+1] = lambda_test1*(test_array1[k]**alpha_test1)*(1 - test_array1[k]**alpha_test1)
        test_array2[k+1] = lambda_test2*(test_array2[k]**alpha_test2)*(1 - test_array2[k]**alpha_test2)

    plt.figure()
    plt.plot(np.arange(0.5, 1, 0.5/149), test_array1[1:], 'bo', np.arange(0.5, 1, 0.5/149), test_array1[1:], 'k', linestyle='default', lw=1.0)
    plt.grid()
    plt.suptitle('lambda=3.6, alpha=0.73')

    plt.figure()
    plt.plot(np.arange(0.5, 1, 0.5/149), test_array2[1:], 'ro', np.arange(0.5, 1, 0.5/149), test_array2[1:], 'k', linestyle='default', lw=1.0)
    plt.grid()
    plt.suptitle('lambda=3.7, alpha=0.73')
    plt.show()

    print('Running test C')
    alpha_array = [0.1, 0.15, 0.2, 0.85, 0.9, 0.95]
    lambda_final = 4
    
    
    val_array_1 = val_array_2 = val_array_3 = val_array_4 = val_array_5 = val_array_6 = np.zeros( 150 )
    val_array_1[0] = val_array_2[0] = val_array_3[0] = val_array_4[0] = val_array_5[0] = val_array_6[0] = start_value
    val_arrays = [val_array_1, val_array_2, val_array_3 ,val_array_4 ,val_array_5, val_array_6]

    for i in range(6):
        for k in range (len(val_array_1) - 1):
            val_arrays[i][k+1] = lambda_final*val_arrays[i][k]**alpha_array[i]*(1 - val_arrays[i][k]**alpha_array[i])
        plt.figure()
        plt.plot(np.arange(0.5, 1, 0.5/149), val_arrays[i][1:], 'bo', np.arange(0.5, 1, 0.5/149), val_arrays[i][1:], 'k', linestyle='default', lw=1.0)
        plt.grid()
        plt.suptitle('lambda=4, alpha=' + str(alpha_array[i]))
    plt.show()

if __name__ == "__main__":
    main()

