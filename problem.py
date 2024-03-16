import numpy as np
import matplotlib.pyplot as plt
import scipy

class Problem:
    def __init__(self, potantal, max_R = 100, precision=1000, effective_mass=1, angular_momentum=1, h_bar=1):
        self.potantal = potantal
        self.r_array = np.linspace(0, max_R, precision)
        self.U_array = np.zeros(len(self.r_array))
        self.W_array = np.zeros(len(self.r_array))
        self.effective_mass = effective_mass
        self.h_bar = h_bar
        self.angular_momentum = angular_momentum
        self.delta = max_R/precision

        self.U_array[0] = 0
        self.U_array[1] = self.delta**(self.angular_momentum + 1)

        
    def get_last_U(self, E):
        for i in range(2, len(self.r_array)):
            self.U_array[i] = self.U_k_plus_1(i, E)
        return self.U_array[-1]
    
    def U_k_plus_1(self, i, E):
        self.W_array[i] = 2*self.effective_mass*(E - self.potantal(self.r_array[i]))/(self.h_bar**2) - self.angular_momentum*(self.angular_momentum + 1)/(self.r_array[i]**2)

        first_term = (2 - (5/6)*self.delta**2*self.W_array[i-1]) * self.U_array[i-1]
        second_term = (1 + (1/12)*self.delta**2*self.W_array[i-2]) * self.U_array[i-2]
        third_term = (1 + (1/12)*self.delta**2*self.W_array[i])

        return (first_term - second_term)/third_term

    def find_E(self, E_guess, max_iterations=1000, tolerance=0.01):
        roots = scipy.optimize.root(self.get_last_U, x0 = E_guess, tol=tolerance)
        return roots