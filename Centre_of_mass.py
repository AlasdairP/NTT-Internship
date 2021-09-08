"""
Idea is to input the solutions of the Wannier problem as a potential landscape
for the CoM problem.

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy import sparse
from scipy import optimize
from scipy.sparse.linalg import eigs,eigsh

# My Files
import Excitons_in_TMDs
import Finite_differences
import Grids_2D


# Constants (AU)
hbar = 1
m0 = 1
e = 1
eps0 = 1/(4*math.pi)

Bohr_to_Ang = constants.value('Bohr radius') *1e10
# 0.529
Hartree_in_eV = constants.value('Hartree energy in eV')
# 27.2
kV_per_cm_to_AU = constants.value('Hartree energy')/(constants.e*constants.value('Bohr radius')) *1e-5
# 5.14e6
Tesla_to_AU = constants.hbar/(constants.e*constants.value('Bohr radius')**2)
# 2.35e5

me = 0.2*m0
mh = 0.5*m0
M = me + mh

# Number of points
Nx = 100
Ny = 100

# Choose grid
exp_grid = False
uniform_grid = True

######################### Create grid #######################

if exp_grid == True:
    
    grid = "Exp. grid"
    delta = 100
    delta_min = 0.4 / Bohr_to_Ang        
    delta_max = 20 / Bohr_to_Ang       
    x,y = Grids_2D.exp_grid(Nx,Ny,delta,delta_min,delta_max)
    
elif uniform_grid == True:
    
    grid = "Uniform grid"
    # System size (input in Angstroms)
    L = 300 / Bohr_to_Ang   *500
    x,y = Grids_2D.uniform_grid(Nx,Ny,L)

[XX, YY] = np.meshgrid(x, y)

RR = (XX**2 + YY**2)**(1/2)

XX_Ang = XX*Bohr_to_Ang
YY_Ang = YY*Bohr_to_Ang

#################### Get FDs for 2nd derivative in Schrodinger Eq ###########

# d^2/dx^2 + d^2/dy^2
diag, diag_p1, diag_pNx, diag_n1, diag_nNx, Lx, Ly = Finite_differences.second_derivative(Nx,Ny,x,y)


# Build potential landscape using energies from Wannier problem code

potential = np.zeros((Nx,Ny))

for i in range(Nx):
    for j in range(Ny):
        
        R_fraction = (RR[i][j]/np.max(RR))*20
        energy_index_to_use = int(np.round(R_fraction,decimals=0))
        if energy_index_to_use > 9:
            potential[i][j] = Excitons_in_TMDs.energy_list0[0]
        else:
            potential[i][j] = Excitons_in_TMDs.energy_list0[10-energy_index_to_use]
# Back to AU
potential = potential/Hartree_in_eV /1000

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
c = ax.pcolormesh(XX_Ang,YY_Ang,potential,shading='auto')
ax.set_title("Potential landscape: purple = "+str(Excitons_in_TMDs.B_min)+" = min. E, yellow = "+str(Excitons_in_TMDs.B_max)+" = larger E")
ax.set_xlabel("x (Ang.)")
ax.set_ylabel("y (Ang.)")
fig.colorbar(c,ax=ax)
plt.show()

############################ Build basic Hamiltonian H_0 ##############

H = sparse.diags(np.ravel(potential))
H += (-hbar**2/(2*M)) *sparse.diags(diag)          
H += (-hbar**2/(2*M)) *sparse.diags(diag_p1, 1)  
H += (-hbar**2/(2*M)) *sparse.diags(diag_pNx, Nx) 
H += (-hbar**2/(2*M)) *sparse.diags(diag_n1, -1) 
H += (-hbar**2/(2*M)) *sparse.diags(diag_nNx, -Nx)
# Periodic BCs
#H += sparse.diags([1], int(Nx*Ny-1))
#H += sparse.diags([1], int(-Nx*Ny+1))


###################### Solve #######################

# Choose number of eigenstates, runs faster for fewer states
k = 4

# Solve (energies returned in meV)
energies,states = Excitons_in_TMDs.symmetrise_and_solve(H,Lx,Ly,k)

############################# Plotting ########################

# Note for E or B fields these are currently the energies for F_max / B_max
print("Binding energies: " + str(energies))

# Probability density
densities = (np.absolute(states))**2  

# Plot colour map of probability density: densities[<state number>]

state_to_plot = k-1

# "unravel" - initially thought (Nx,Ny), but this works
density = densities[state_to_plot].reshape((Ny,Nx))  

fig3 = plt.figure()
ax = fig3.add_subplot(1,1,1)
c = ax.pcolormesh(XX_Ang,YY_Ang, density, shading='auto')
#c = ax.pcolormesh(density)
ax.set_title("Density: 1s," + grid)
ax.set_xlabel("x (Ang.)")
ax.set_ylabel("y (Ang.)")
fig3.colorbar(c,ax=ax)
fig3.tight_layout()
plt.show()









