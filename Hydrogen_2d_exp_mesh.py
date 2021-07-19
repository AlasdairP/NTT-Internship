"""
2D H atom on exp mesh 

"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import constants
from scipy import sparse
from scipy.sparse.linalg import eigs,eigsh
 
# Number of grid points
Nx = 100
Ny = 100

def exp_grid(Nx, Ny, delta,delta_min,delta_max):

    x = np.zeros(Nx)
    y = np.zeros(Ny)
    
    for i in range(int(Nx/2),Nx):
        
        # Separation as per Peeters 2015
        hx_i = delta_min + delta_max*(1-math.exp(-(i - Nx/2)/delta))
        x[i] = x[i-1] + hx_i                            
        
        # Reflect to negative x values: [N/2,N] -> [0,N/2]
        x[Nx-i-1] = -x[i]
        
    for i in range(int(Ny/2),Ny):
        
        # Separation as per Peeters 2015
        hy_i = delta_min + delta_max*(1-math.exp(-(i - Ny/2)/delta))                            
        y[i] = y[i-1] + hy_i
        
        # Reflect to negative y values: [N/2,N] -> [0,N/2]
        y[Ny-i-1] = -y[i]
     
    for i in range(int(Nx/2),Nx):
        x[i] = x[i] - delta_min/2
    for i in range(0,int(Nx/2)):
        x[i] = x[i] + delta_min/2
        
    for i in range(int(Ny/2),Ny):
        y[i] = y[i] - delta_min/2
    for i in range(0,int(Ny/2)):
        y[i] = y[i] + delta_min/2
      
    #L = x[Nx-1] - x[0]
    #x_uniform = np.linspace(-L/2, L/2, Nx)
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot(x,x_uniform,'o')
    #plt.show()
        
    return x,y


# Set d for soft-core
def Coulomb(r, d=0.0):
    Coulomb = -e**2/(4*math.pi*eps0*np.sqrt(r**2 + d**2))
    return Coulomb
         
def finite_differences(Nx,Ny,x,y):
    
    hx = np.zeros(Nx)
    hy = np.zeros(Ny)
    
    for i in range(1,Nx):  
        hx[i] = x[i] - x[i-1]

    for i in range(1,Ny):
        hy[i] = y[i] - y[i-1]
    
    hx[0] = hx[1] + (hx[1] - hx[2])
    hy[0] = hy[1] + (hy[1] - hy[2])
        
    #index = np.arange(Nx)
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot(index,hx,'o')
    #ax.set_title("hx vs index")
    #plt.show()
    
    # Total size of new H matrix
    N = Nx*Ny
    diag = np.zeros(N)
    diag_p1 = np.zeros(N)  
    diag_n1 = np.zeros(N)     
    diag_pNx = np.zeros(N) 
    diag_nNx = np.zeros(N) 
    
    # Make arrays for these so they can be used for symmetrisation later
    # Peeters calls these L^2 
    Lx = np.zeros(N)
    Ly = np.zeros(N)
    
    # Rows
    for j in range(Ny):
        # Columns
        for i in range(Nx):
        
            # Map to single label
            l = j*Nx + i
                
            # Final column
            if i == (Nx-1):
                
                Lx[l] = Lx[l-1]
                Ly[l] = Ly[l-1]
                
                diag[l] = diag[l-1]
                diag_p1[l] = diag_p1[l-1]   
                diag_n1[l] = diag_n1[l-1]    
                diag_pNx[l] = diag_pNx[l-1]
                diag_nNx[l] = diag_nNx[l-1]
        
            # Final row
            elif j == (Ny-1):
                
                Lx[l] = Lx[l-Nx]
                Ly[l] = Ly[l-Nx]
                
                diag[l] = diag[l-Nx]
                diag_p1[l] = diag_p1[l-Nx]   
                diag_n1[l] = diag_n1[l-Nx]    
                diag_pNx[l] = diag_pNx[l-Nx] # Final row deleted later anyway
                diag_nNx[l] = diag_nNx[l-Nx]  
            
            # The majority of points. 
            # Having i+1 and j+1 is ok as last row/col not included here
            else:
                
                Lx[l] = (hx[i+1]+hx[i])/2
                Ly[l] = (hy[j+1]+hy[j])/2
                
                diag[l] =  -2/(hx[i]*hx[i+1])
                diag[l] += -2/(hy[j]*hy[j+1])
                
                diag_p1[l] =  1/(hx[i+1] *Lx[l])
                diag_n1[l] =  1/(hx[i]   *Lx[l]) 
                diag_pNx[l] = 1/(hy[j+1] *Ly[l])
                diag_nNx[l] = 1/(hy[j]   *Ly[l])
       
    # Choose a set of FDs to plot            
    #FD_toplot = diag.reshape((Nx,Ny))
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #c = ax.pcolormesh(XX,YY, FD_toplot, shading='auto')
    #ax.set_title("FD")
    #fig.colorbar(c,ax=ax)
    #fig.tight_layout()
    #plt.show()
             
    # delete final/first entry to fit into H matrix
    diag_p1 = np.delete(diag_p1, [N-1])
    diag_n1 = np.delete(diag_n1, [0])
    
    # delete final/first row to fit into H matrix
    diag_pNx = np.delete(diag_pNx, [N - i for i in range(1,Nx+1)])
    diag_nNx = np.delete(diag_nNx, [i for i in range(Nx)])
    
    return diag, diag_p1, diag_pNx, diag_n1, diag_nNx, Lx, Ly
        
      
# Constants
hbar = constants.hbar
m = constants.m_e  
e = constants.e
eps0 = constants.epsilon_0      
 
    
# Grid spacing values used in Peeters 2015: delta_min = 5e-5 Angstroms etc.

delta = 100
delta_min = 0.03e-10     # Adjust for Hydrogen - smaller radii than excitons  
delta_max = 1.5e-10  

# note: The spacing at the edge isn't actually delta_max. It is delta_max + delta_min
# which is obviously very close to delta_max when there is a large difference between the two. 

# Create exponentially scaled grid
x,y = exp_grid(Nx,Ny,delta,delta_min,delta_max)
[XX, YY] = np.meshgrid(x, y)

# Calculate r values and get potential
r = (x**2 + y**2)**(1/2)
RR = (XX**2 + YY**2)**(1/2)
V = Coulomb(RR)

# Get finite difference coefficients
diag, diag_p1, diag_pNx, diag_n1, diag_nNx, Lx, Ly = finite_differences(Nx,Ny,x,y)

# Construct Hamiltonian

H = sparse.diags((np.ravel(V, order='C'))) # 'C' is default, 'K' is opposite
H += (-hbar**2/(2*m)) *sparse.diags(diag)                
H += (-hbar**2/(2*m)) *sparse.diags((diag_p1), 1)
H += (-hbar**2/(2*m)) *sparse.diags((diag_pNx), Nx) 
H += (-hbar**2/(2*m)) *sparse.diags((diag_n1), -1) 
H += (-hbar**2/(2*m)) *sparse.diags((diag_nNx), -Nx)                 

######### Symmetrisation following Peeters #######

# I'm unsure how this symmetrises H since L and L_inv are diagonal
Lx = np.sqrt(Lx)
Ly = np.sqrt(Ly)

L = Lx*Ly  

N = Nx*Ny
L_inv = np.zeros(N)
for l in range(N):
    L_inv[l] = 1/L[l]
    
L = sparse.diags(L)
L_inv = sparse.diags(L_inv)

H = L_inv*L*L*H*L_inv    


# Find lowest k eigenvalues
k=6
energies,states = eigsh(H, k=k, which='SA')
#energies, states = eig(H, k=k, which='SM')

# electron volts
energies = energies/constants.eV

# Ordered
states = np.array([x for _, x in sorted(zip(energies, states.T), key=lambda pair: pair[0])])
energies = np.sort(energies)
print(energies)

# Prob. density
densities = (np.absolute(states))**2 # *4*math.pi*np.ravel(RR) # (or *2*pi*np.ravel(RR))?

########## Plotting ##############
# Angstroms
r = r*1e10
XX = XX*1e10
YY = YY*1e10
x = x*1e10
y = y*1e10

# Choose starting eigenstate (will plot i, i+1, i+2, i+3)
i = 0
"""
energy0 = energies[i]
energy1 = energies[i+1]
energy2 = energies[i+2]
energy3 = energies[i+3]

# Plot the lowest four states

xtext = 2
ytext = 0.00002

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
#ax2.set_ylim(-50,+10)
ax2.set_title("Energy eigenvalues")
plt.show()
"""

################## Plot colour map ########################

# "Unravel"
density = densities[k-1].reshape((Nx,Ny))

fig3 = plt.figure()
ax = fig3.add_subplot(1,1,1)
c = ax.pcolormesh(XX,YY, density.T, shading='auto')
ax.set_title("(2,0,0) exp with symmetrisation")
fig3.colorbar(c,ax=ax)
fig3.tight_layout()
plt.show()

"""
            # Peeters
            diag[l] += -1/(hy[i+1]*Lyi)
            diag[l] += -1/(hy[i]  *Lyi)
            
            diag_p1[l] =  1/(hy[i+1]*Lyi)
            diag_n1[l] =  1/(hy[i]  *Lyi) 
            diag_pNy[l] = 1/(hx[j+1]*Lxj)
            diag_nNy[l] = 1/(hx[j]  *Lxj)
"""
