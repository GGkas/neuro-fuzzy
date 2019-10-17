import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

#pylint: disable=no-member
def main():
  p1 = [-1.0, -1.0]
  p2 = [0, 0]
  p3 = [-1.0, 1.0]

  p_vector = [p1, p2, p3]
  target_vector = [0, 0, 1]

  weight = [1, 0]
  bias = 0.5

  epsilon = 1E-3

  epoch = 1
  falses = 1

  while falses > 0:
    falses = 0
    for i in range(len(p_vector)):
      a = sp.expit(np.dot(p_vector[i], weight) + bias)
      error = target_vector[i] - a
      if (abs(error) > epsilon):
        falses += 1
        bias += error
        weight += np.array(p_vector[i])*error
        print('\n' + str(bias) + ' ' + str(weight) + '\n')
    epoch += 1
    print('\n\nEpoch ' + str(epoch) + ' finished, with ' + str(falses) + ' faults.')

  print('Final bias: ' + str(bias))
  print('Final weight: ' + str(weight))
  
  x = np.arange(-4, 4, 0.2)
  plt.figure()
  plt.plot(x, -(weight[0]/weight[1])*x - bias/weight[1], 'k', p1[0], p1[1], 'b^', p2[0], p2[1], 'b^', p3[0], p3[1], 'b*')
  plt.title('Decision boundary for initial weight of (1 0) and initial bias of 0.5')
  plt.grid()
  plt.axhline(linewidth='0.5', color='r')
  plt.axvline(linewidth='0.5', color='r')
  plt.show()


if __name__ == "__main__":
  main()