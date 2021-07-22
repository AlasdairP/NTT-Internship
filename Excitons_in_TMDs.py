"""
Master. Working in AU.
Choose uniform or exponential 2D grid.
Choose (screened) Coulomb or Keldysh potential.


"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import constants
from scipy import sparse
from scipy.sparse.linalg import eigs,eigsh

# My files
import Grids_2D
import Finite_differences
import Potentials

def symmetrise_and_solve(H,Lx,Ly,k):
    
    # Symmetrise 
    
    Lx = np.sqrt(Lx)
    Ly = np.sqrt(Ly)
    L = Lx*Ly  
    L_inv = 1/L    
    L = sparse.diags(L)
    L_inv = sparse.diags(L_inv)
    H = L_inv*L*L*H*L_inv
    
    # Solve Schrodinger Eq 
    
    energies,states = eigsh(H, k=k, which='SA')
    #energies,states = eigs(H, k=k, which='SR')
    
    # Order
    states = np.array([st for _, st in sorted(zip(energies, states.T), key=lambda pair: pair[0])])
    energies = np.sort(energies)
    
    # Hartree AU -> meV
    energies = energies*27.2114 *1000
    
    return energies,states

######################## Main ###############################

# Constants (AU)
hbar = 1
m0 = 1
e = 1
eps0 = 1/(4*math.pi)

######################## Get material/system params ##########
# from text file?
eps = 13 * eps0
eps1 = 1.0 * eps0
eps2 = 1.0 * eps0
mu = 0.15*m0
rho0 = 10.79 / 0.529177

######################### Create grid #######################

# Number of points
Nx = 100
Ny = 100

exp_grid = True
uniform_grid = False

if exp_grid == True:
    
    grid = "Exp. grid"
    delta = 100
    delta_min = 1 / 0.529177        
    delta_max = 30 / 0.529177       
    
    x,y = Grids_2D.exp_grid(Nx,Ny,delta,delta_min,delta_max)
    
if uniform_grid == True:
    
    grid = "Uniform grid"
    # System size (input in Angstroms)
    L = 300 / 0.529177
    x,y = Grids_2D.uniform_grid(Nx,Ny,L)

[XX, YY] = np.meshgrid(x, y)

#################### Get FDs for 2nd derivative in Schrodinger Eq ###########

# d^2/dx^2 + d^2/dy^2
diag, diag_p1, diag_pNx, diag_n1, diag_nNx, Lx, Ly = Finite_differences.second_derivative(Nx,Ny,x,y)

############################## Get potential ##################

# Calculate r values 
RR = (XX**2 + YY**2)**(1/2)

# Choose soft-core d value in Bohr radii
d = 0.01

Coulomb = False
Keldysh = True

if Coulomb == True:
    V = Potentials.Coulomb(RR,eps,d)
    
if Keldysh == True:
    V = Potentials.Keldysh(RR,eps1,eps2,rho0,d)
    
    # fudge factor - still not 100% sure on this but it gives the right numbers
    V = V / 27.2114
    
############################ Build basic Hamiltonian H_0 ##############

H = sparse.diags(np.ravel(V))
H += (-hbar**2/(2*mu)) *sparse.diags(diag)          
H += (-hbar**2/(2*mu)) *sparse.diags(diag_p1, 1)  
H += (-hbar**2/(2*mu)) *sparse.diags(diag_pNx, Nx) 
H += (-hbar**2/(2*mu)) *sparse.diags(diag_n1, -1) 
H += (-hbar**2/(2*mu)) *sparse.diags(diag_nNx, -Nx)
    
############################ Add external E or B field and Solve SE #############

Electric = False
Magnetic = False

# Choose number of eigenstates, runs faster for fewer states
k = 6
    
if Electric == True:
    
    # Choose range

    F_min = -100   # input in KV/cm
    F_max = 100
    number = 20
    F = np.linspace(F_min, F_max, number)
    F = F / 5.1422e6  # -> AU

    energy_list = np.zeros(number)
    
    for i, F_strength_i in enumerate(F):
        
        E = - e * F_strength_i * XX
        H += sparse.diags(np.ravel(E))
        
        # Solve (energies returned in meV)
        energies,states = symmetrise_and_solve(H,Lx,Ly,k)

        GS_energy = energies[0]
        energy_list[i] = GS_energy
        print(GS_energy)
        
        # Remove this E field from H ready for the next one
        H -= sparse.diags(np.ravel(E))
        
    """
    # Complex absorbing potential - a la Gawlas 2019
    W = RR**6
    
    # scaling param - specific to eigenstate and E field strength
    eta = 1e38   
    
    H_CAP = 1j * eta * W
    H += sparse.diags(np.ravel(H_CAP))
    """
    
elif Magnetic == True:
    
    # field strength - input in Tesla
    B_max = 10
    B_min = -10
    number = 20
    B = np.linspace(B_min,B_max,number)
    B = B / 2.3505176e5  # -> AU
    
    energy_list = np.zeros(number)
    
    for i, B_i in enumerate(B):
            
        B_term1 = (1/(2*mu)) * e**2 * B_i**2 * XX**2
        B_term2_pref = np.ravel((hbar/(mu)) * 1j * e * B_i)
        
        diag, diag_nNx, diag_pNx = Finite_differences.first_derivative_y(Nx,Ny,y)
        
        B_diag = B_term2_pref * diag
        B_diag_nNx = B_term2_pref * diag_nNx
        B_diag_pNx = B_term2_pref * diag_pNx
        
        # Construct Hamiltonian
        
        H += sparse.diags(np.ravel(B_term1))
        H += sparse.diags(B_diag)
        H += sparse.diags(B_diag_nNx, -Nx)  
        H += sparse.diags(B_diag_pNx, +Nx) 
        
        # Solve (energies returned in meV)
        energies,states = symmetrise_and_solve(H,Lx,Ly,k)
        
        GS_energy = energies[0]
        energy_list[i] = GS_energy
        print(GS_energy)
        
        # Remove ready for next iteration
        H -= sparse.diags(np.ravel(B_term1))
        H -= sparse.diags(B_diag)
        H -= sparse.diags(B_diag_nNx, -Nx)  
        H -= sparse.diags(B_diag_pNx, +Nx)

else:
    
    # Energies returned in meV    
    energies,states = symmetrise_and_solve(H,Lx,Ly,k)

############################# Plotting ########################

# Note for E or B fields these are currently the energies for E_max / B_max
print("Binding energies: " + str(energies))

# Probability density
densities = (np.absolute(states))**2  

# Back to Angstroms
XX = XX*0.529177
YY = YY*0.529177
x = x  *0.529177
y = y  *0.529177

if Electric == True:
    # Plot GS energy against E field
    
    # Back to kV/cm
    F = F * 5.1422e6
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(F,energy_list, 'o')
    ax.set_title("Stark shift, " + grid)
    ax.set_xlabel("E field strength (kV/cm)")
    ax.set_ylabel("Ground state binding energy (meV)")
    fig.tight_layout()
    plt.show()

elif Magnetic == True:
    
    # GS energy against B field
    
    # Back to Tesla
    B = B * 2.3505176e5
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(B,energy_list, 'o')
    ax.set_title("Magnetic field effect, " + grid)
    ax.set_xlabel("B field strength (T)")
    ax.set_ylabel("Ground state binding energy (meV)")
    fig.tight_layout()
    plt.show()


# Plot colour map of probability density: densities[<state number>]

state_to_plot = 0

# "unravel" - initially thought (Nx,Ny), but this works
density = densities[state_to_plot].reshape((Ny,Nx))  

fig3 = plt.figure()
ax = fig3.add_subplot(1,1,1)
c = ax.pcolormesh(XX,YY, density, shading='auto')
ax.set_title("n = " + str(state_to_plot) + ", " + grid)
ax.set_xlabel("x (Ang.)")
ax.set_ylabel("y (Ang.)")
fig3.colorbar(c,ax=ax)
fig3.tight_layout()
plt.show()
