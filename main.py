import numpy as np
import matplotlib.pyplot as plt
from problem import Problem




# working in ev
h_bar = 6.5821220E-16
mass = 938.272 * 10**6
fine_structure = 1/137
bohr_radius = 2.68 * (10**-4)
light_speed = 299792458

def potential(r):

    return fine_structure*h_bar*light_speed/r

problem = Problem(
        potential,
        angular_momentum = 1/2,
        max_R = bohr_radius,
        precision = 10000,
        effective_mass = mass,
        h_bar=h_bar,
    )

E = problem.find_E(0)
print(E)

plt.plot(problem.r_array, problem.U_array)
plt.show()

