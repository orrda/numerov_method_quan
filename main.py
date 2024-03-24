import numpy as np
import matplotlib.pyplot as plt
from problem import Problem


hbarc = 197.3269804 #fm*MeV
hbar = 6.582119569*(10**-22) #MeV*s
fine_structure_constant = 0.0072973525693
c = hbarc/hbar #fm/s

nucleon_mass = 931.49432/(c**2) #MeV*s^2/fm^2

hydrogen_charge = 1
hydrogen_nucleon_num = 1
hydrogen_nucleus_radius = 0.88 #fm
hydrogen_mass = hydrogen_nucleon_num*nucleon_mass

deuterium_charge = 1
deuterium_nucleon_num = 2
deuterium_nucleus_radius = 2.14 #fm
deuterium_mass = deuterium_nucleon_num*nucleon_mass

helium_charge = 2
helium_nucleon_num = 4
helium_nucleus_radius = 1.97 #fm
helium_mass = helium_nucleon_num*nucleon_mass

kaon_charge = -1
kaon_mass = 493.7/(c**2)

reduced_mass = hydrogen_mass*kaon_mass/(hydrogen_mass+kaon_mass)
bohr_radius = hbarc/(hydrogen_charge*fine_structure_constant*reduced_mass*(c**2))
rydberg_energy = 0.5*(fine_structure_constant**2)*(hydrogen_charge**2)*reduced_mass*(c**2)

kaon_charge = -1
kaon_mass = 493.7/(c**2)

def potential(r):
    return -1*fine_structure_constant*hydrogen_charge*hbarc/r

problem = Problem(
        potential,
        angular_momentum = 0,
        max_R = bohr_radius*10,
        precision = 100000,
        effective_mass = reduced_mass,
        h_bar=hbar,
    )

# question 3.1.2

arr = np.array([-1.1, -1.05, -1, -0.95, -0.9])
arr = arr*rydberg_energy
for e in arr:
    U_array = problem.get_U(e)
    normalized_U = U_array/sum(np.abs(U_array))
    plt.plot(problem.r_array, normalized_U)


plt.xlabel('r [fm]')
plt.ylabel('psi(r)')
plt.title('U(r) for different E')
plt.grid(True)
plt.legend([-1.1, -1.05, -1, -0.95, -0.9])
plt.show()


# question 3.1.4

E_guess = - rydberg_energy

problem = Problem(
        potential,
        angular_momentum = 0,
        max_R = bohr_radius*20,
        precision = 100,
        effective_mass = reduced_mass,
        h_bar=hbar,
    )

N_array = np.linspace(2, 5, 4)
N_array = 10**N_array
eta_array = []

for N in N_array:
    problem = Problem(
        potential,
        angular_momentum = 0,
        max_R = bohr_radius*20,
        precision = int(N),
        effective_mass = reduced_mass,
        h_bar=hbar,
    )
    root = problem.find_E(E_guess, tolerance=0.0001)
    eta = 1 - root/-rydberg_energy
    eta_array.append(eta)
    print(N, eta)

plt.plot(N_array, eta_array, 'o-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('eta')
plt.title('Convergence of eta')
plt.grid(True)
plt.show()



# question 3.1.5

E_guess = - rydberg_energy


for n in range(1, 4):
    potential = lambda r: -1*fine_structure_constant*hydrogen_charge*hbarc/r
    for l in range(0, n):
        problem = Problem(
            potential,
            angular_momentum = l,
            max_R = bohr_radius*20*n,
            precision = 20000,
            effective_mass = reduced_mass,
            h_bar=hbar,
        )
        E_guess = -rydberg_energy/n**2
        root = problem.find_E(E_guess, tolerance=10**-6)
        U_array = problem.get_U(root)
        normalized_U = U_array/sum(np.abs(U_array))
        plt.plot(problem.r_array, normalized_U)
        eta = 1 - root/(-rydberg_energy/n**2)
        avr = np.sqrt(problem.get_average(root))


        print("n=",n,", l=", l,", energy=", root,", eta=", eta,", average r=", avr)

plt.xlabel('r [fm]')
plt.ylabel('psi(r)')
plt.title('U(r) for different l')
plt.grid(True)
plt.legend(['n=1, l=0', 'n=2, l=0', 'n=2, l=1', 'n=3, l=0', 'n=3, l=1', 'n=3, l=2'])
plt.show()

# question 3.2.3

V_0 = np.pi*(bohr_radius**3)*283*(10**-6)
potential = lambda r: -1*fine_structure_constant*hydrogen_charge*hbarc/r + V_0*np.exp(-r**2/(2*hydrogen_nucleus_radius**2))/((np.sqrt(2*np.pi)*hydrogen_nucleus_radius)**3)


for n in range(1,4):
    for l in range(0, n):
        problem = Problem(
            potential,
            angular_momentum = l,
            max_R = bohr_radius*20*n,
            precision = 20000,
            effective_mass = reduced_mass,
            h_bar=hbar,
        )
        E_guess = -rydberg_energy/n**2
        root = problem.find_E(E_guess, tolerance=10**-6)
        U_array = problem.get_U(root)
        normalized_U = U_array/sum(np.abs(U_array))
        plt.plot(problem.r_array, normalized_U)
        eta = 1 - root/(-rydberg_energy/n**2)
        avr = np.sqrt(problem.get_average(root))


        print("n=",n,", l=", l,", energy=", root,", eta=", eta,", average r=", avr)

plt.xlabel('r [fm]')
plt.ylabel('psi(r)')
plt.title('U(r) for different l')
plt.grid(True)
plt.legend(['n=1, l=0', 'n=2, l=0', 'n=2, l=1', 'n=3, l=0', 'n=3, l=1', 'n=3, l=2'])
plt.show()
