import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import warnings as warn

#####################################################################################
# The update equation for vlbp:                                                     #
#   x(k+1) = [(1+gamma)I - (1-gamma)aH]x(k) - gamma*x(k-1) - (1-gamma)a*d.          #
#   On first iteration, x(k-1) is not defined, so it's not part of the equation.    #
#####################################################################################

def func_value(H, d, constant, x1, x0):
    return [0.5*np.dot(x0.T, np.matmul(H, x0)) + np.dot(d, x0) + constant, 0.5*np.dot(x1.T, np.matmul(H, x1)) + np.dot(d, x1) + constant] 

def vlbp_training(a, gamma_orig, heta, ro, z, x_guess, H, d, c, iteration):
    # epsilon = 1E-3

    loc_gamma = gamma_orig

    for _ in range(x_guess.shape[0] - 1):
        if (iteration == 0):
            x_guess[iteration+1] = np.matmul(((1+loc_gamma)*np.identity(2) - (1-loc_gamma)*a*H), x_guess[iteration]) - (1-loc_gamma)*a*d
        else:
            x_guess[iteration+1] = np.matmul(((1+loc_gamma)*np.identity(2) - (1-loc_gamma)*a*H), x_guess[iteration]) - loc_gamma*x_guess[iteration-1] - (1-loc_gamma)*a*d
        vals = func_value(H, d, c, x_guess[iteration + 1], x_guess[iteration])
        if ((vals[1]-vals[0])/vals[0] > z):
            x_guess[iteration+1] = x_guess[iteration]
            a *= ro
            loc_gamma = 0
            iteration+=1
            continue
        elif ((vals[1]-vals[0])/vals[0] < 0):
            a *= heta
            if(loc_gamma == 0):
                loc_gamma = gamma_orig
            iteration += 1
            continue
        else:
            if(loc_gamma == 0):
                loc_gamma = gamma_orig
            iteration += 1
    
    return x_guess


def main():
    
    # F(x) parameters
    H_mtx = np.array([[3, 1], [1, 3]])
    d_vec = np.array([1, 2])
    constant = 2

    # Training parameters
    alpha = 0.4
    gamma = 0.1
    heta = 1.5
    ro = 0.5
    zeta = 5

    # Initial guess
    guesses = np.zeros((4, 2))
    guesses[0] = np.array([-1, -2.5])
    iteration = 0

    guess_array = vlbp_training(alpha, gamma, heta, ro, zeta, guesses, H_mtx, d_vec, constant, iteration)

    # Plotting
    x1, x2 = sym.symbols('x1 x2')
    func = sym.lambdify((x1, x2), 1.5*x1**2 + 1.5*x2**2 + x1*x2 + x1 + 2*x2 + 2, 'numpy')
    xaxis = np.linspace(-5, 5, 1000)
    yaxis = np.linspace(-6, 6, 1000)
    x, y = np.meshgrid(xaxis, yaxis)
    plt.contour(x, y, func(x, y))
    plt.margins(-0.25, 0.25)
    with warn.catch_warnings():
        warn.simplefilter("ignore")
        plt.plot(guess_array[:, 0], guess_array[:, 1], 'g^', guess_array[:, 0], guess_array[:, 1], 'k', guess_array[0][0], guess_array[0][1], 'b*', linestyle='default', lw=1.0)
    plt.grid()
    plt.axhline(linewidth='0.5', color='r')
    plt.axvline(linewidth='0.5', color='r')
    plt.show()

if __name__ == "__main__":
    main()