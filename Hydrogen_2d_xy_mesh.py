"""
Trying to extend the standard H atom to an xy grid 

"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import constants
from scipy import sparse
from scipy.sparse.linalg import eigsh
 
# Number of grid points
Nx = 200
Ny = 200

# Total size of system
L = 2e-9  

# Step size
hx = L/Nx
hy = L/Ny


def Coulomb(r, d=hx):
    Coulomb = -e**2/(4*math.pi*eps0*np.sqrt(r**2 + d**2))
    return Coulomb
   
def finite_differences(Nx,Ny,hx,hy):
    
    # Following Peeters (5) with factor of -1/mu removed
    
    diag = -2/(hx**2) *np.ones(Nx*Ny)                # 4 terms from Peeters (5) top line
    diag += -2/(hy**2) *np.ones(Nx*Ny) 
    off_diag_1 = 1/(hx**2) *np.ones((Nx*Ny)-1)       # Peeters (5) 2nd line
    off_diag_Nx = 1/(hy**2) *np.ones((Nx*Ny)-Nx)     # Peeters (5) 3rd line

    return diag,off_diag_1,off_diag_Nx
        
      
# Constants
hbar = constants.hbar
m = constants.m_e
e = constants.e
eps0 = constants.epsilon_0        
 
    
x = np.linspace(-L/2, L/2-hx, Nx)
y = np.linspace(-L/2, L/2-hy, Ny)

[XX, YY] = np.meshgrid(x, y)

# Calculate r values and get potential

r = (x**2 + y**2)**(1/2)
RR = (XX**2 + YY**2)**(1/2)
V = Coulomb(RR)

# Add uniform E field
#E = -1e-11 * YY


# Get finite difference coefficients
diag, off_diag_1, off_diag_Nx = finite_differences(Nx,Ny,hx,hy)

# Construct Hamiltonian

#H =  np.zeros([Nx*Ny,Nx*Ny])

H = sparse.diags((np.ravel(V)))
#H += sparse.diags((np.ravel(E)))
H += (-hbar**2/(2*m)) *sparse.diags(diag)                
H += (-hbar**2/(2*m)) *sparse.diags((off_diag_1), 1)
H += (-hbar**2/(2*m)) *sparse.diags((off_diag_Nx), Nx) 
H += (-hbar**2/(2*m)) *sparse.diags((off_diag_1), -1) 
H += (-hbar**2/(2*m)) *sparse.diags((off_diag_Nx), -Nx)                 

# Find lowest k eigenvalues
k=5
energies,states = eigsh(H, k=k, which='SA')
#energies,states = np.linalg.eig(H)

# electron volts
energies = energies/constants.eV

# Ordered
states = np.array([x for _, x in sorted(zip(energies, states.T), key=lambda pair: pair[0])])
energies = np.sort(energies)
print(energies)

# Prob. density
densities = (np.absolute(states))**2  *4*math.pi*np.ravel(RR) #for radial prob density I think?

# Choose starting eigenstate (will plot i, i+1, i+2, i+3)
i = 0

# Angstroms
r_plot = r*1e10
XX = XX*1e10
YY = YY*1e10
x = x*1e10
"""
energy0 = energies[i]
energy1 = energies[i+1]
energy2 = energies[i+2]
energy3 = energies[i+3]

# Plot the lowest four states

xtext = 2
ytext = 0.0

fig1 = plt.figure()
ax = fig1.add_subplot(2,2,1)
ax.plot(x,densities[i].reshape((Nx,Ny))[int(Nx/2)],'bo',markersize=2)
ax.set_xlabel("r (Ang)")
ax.set_ylabel("Wavefunction")
ax.set_title("n = 1")
#ax.set_ylim(-0.01,+0.1)
ax.text(xtext,ytext,("E = " +str(round(energy0,4))+ " eV"))

ax2 = fig1.add_subplot(2,2,2)
ax2.plot(x,densities[i+1].reshape((Nx,Ny))[int(Nx/2)],'bo',markersize=2)
ax2.set_xlabel("r (Ang)")
ax2.set_ylabel("Wavefunction")
ax2.set_title("n = 2")
#ax2.set_ylim(-0.01,+0.1)
ax2.text(xtext,ytext,("E = " +str(round(energy1,4))+ " eV"))

ax3 = fig1.add_subplot(2,2,3)
ax3.plot(x,densities[i+2].reshape((Nx,Ny))[int(Nx/2)],'bo',markersize=2)
ax3.set_xlabel("r (Ang)")
ax3.set_ylabel("Wavefunction")
ax3.set_title("n = 3")
#ax3.set_ylim(-0.01,+0.1)
ax3.text(xtext,ytext,("E = " +str(round(energy2, 4))+ " eV"))

ax4 = fig1.add_subplot(2,2,4)
ax4.plot(x,densities[i+3].reshape((Nx,Ny))[int(Nx/2)],'bo',markersize=2)
ax4.set_xlabel("r (Ang)")
ax4.set_ylabel("Wavefunction")
ax4.set_title("n = 4")
#ax4.set_ylim(-0.01,+0.1)
ax4.text(xtext,ytext,("E = " +str(round(energy3,4))+ " eV"))
plt.tight_layout()
plt.show()

# Plot energies of states against state index

number_of_states = k
state_indices = np.arange(number_of_states)
energy_list = np.zeros(number_of_states)
for i in range(number_of_states):
    energy_list[i] = energies[i]

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(state_indices,energy_list)
ax2.set_xlabel("State index")
ax2.set_ylabel("Energy (eV)")
ax2.set_ylim(-50,+10)
ax2.set_title("Energy eigenvalues")
plt.show()
"""
################## Plot colour map ########################

# "Unravel"

#density = np.unravel_index(densities[0], (100, 100))

density = np.zeros((Nx,Ny))    
for j in range(Ny):
    for i in range(Nx):
        
        l = Nx*j + i
        density[i][j] = densities[k-1][l] 

fig3 = plt.figure()
ax = fig3.add_subplot(1,1,1)
c = ax.pcolormesh(XX, YY, density, shading='auto')
ax.set_title("(3,2,0)")
fig3.colorbar(c,ax=ax)
fig3.tight_layout()
plt.show()
