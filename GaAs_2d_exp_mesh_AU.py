"""
GaAs, 2D exp mesh, AU, Coulomb or Keldysh

"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import special
from scipy import constants
from scipy import sparse
from scipy.sparse.linalg import eigs,eigsh
 
# Number of grid points
Nx = 100
Ny = 100

def exp_grid(Nx, Ny, delta,delta_min,delta_max):

    # Create non-uniform mesh of points (x,y)
    x = np.zeros(Nx)
    y = np.zeros(Ny)
    
    for i in range(int(Nx/2),Nx):
        
        # Separation as per Peeters 2015
        hx_i = delta_min + delta_max*(1-math.exp(-(i - Nx/2)/delta)) # removed -1 
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


def Coulomb(r, d=0.0):
    Coulomb = -e**2/(4*math.pi*eps*np.sqrt(r**2 + d**2))
    return Coulomb

# Keldysh potential

def Keldysh(r,eps1,eps2,rho0,d=0.0):
     
    # Zeroth order Struve function
    H0 = special.struve(0,(np.sqrt(r**2 + d**2)/rho0))
    
    # Neumann function
    Y0 = special.y0(np.sqrt(r**2 + d**2)/rho0)
    
    # Keldysh
    prefactor = -2*math.pi/((eps1+eps2)*rho0)
    
    V = prefactor*(H0 - Y0) 
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(RR,V,'o')
    plt.show()
    return V
   
         
def finite_differences(Nx,Ny,x,y):
    
    # Following Peeters (5) with factor of -1/mu removed
    
    hx = np.zeros(Nx)
    hy = np.zeros(Ny)
    
    for i in range(1,Nx):  
        hx[i] = x[i] - x[i-1]

    for i in range(1,Ny):
        hy[i] = y[i] - y[i-1]
    
    # First one is linearly spaced ...negligible error
    hx[0] = hx[1] + (hx[1] - hx[2]) 
    hy[0] = hy[1] + (hy[1] - hy[2]) 
    
    N = Nx*Ny
    
    diag = np.zeros(N)
    diag_p1 = np.zeros(N)
    diag_n1 = np.zeros(N)    
    diag_pNx = np.zeros(N) 
    diag_nNx = np.zeros(N) 
    
    Lx = np.zeros(N)
    Ly = np.zeros(N)
    
    for j in range(Ny):
        for i in range(Nx):
        
            l = j*Nx + i
            
            if i == (Nx-1):
                Lx[l] = Lx[l-1]
                Ly[l] = Ly[l-1]
                diag[l] = diag[l-1]
                diag_p1[l] = diag_p1[l-1]   
                diag_n1[l] = diag_n1[l-1]    
                diag_pNx[l] = diag_pNx[l-1]
                diag_nNx[l] = diag_nNx[l-1]
            elif j == (Ny-1):
                Lx[l] = Lx[l-Nx]
                Ly[l] = Ly[l-Nx]
                diag[l] = diag[l-Nx]
                diag_p1[l] = diag_p1[l-Nx]   
                diag_n1[l] = diag_n1[l-Nx]    
                diag_pNx[l] = diag_pNx[l-Nx] # Don't need final row - wil delete anyway
                diag_nNx[l] = diag_nNx[l-Nx]                
                
            else:
            
                Lx[l] = (hx[i+1]+hx[i])/2
                Ly[l] = (hy[j+1]+hy[j])/2
                
                diag[l] =  -2/(hx[i]*hx[i+1])      
                diag[l] += -2/(hy[j]*hy[j+1])      
                
                diag_p1[l] =  1/(hx[i+1]*Lx[l])   
                diag_n1[l] =  1/(hx[i]  *Lx[l])   
                diag_pNx[l] = 1/(hy[j+1]*Ly[l])   
                diag_nNx[l] = 1/(hy[j]  *Ly[l])  
    
    diag_p1 = np.delete(diag_p1, [N-1])
    diag_n1 = np.delete(diag_n1, [0])
    
    diag_pNx = np.delete(diag_pNx, [N - i for i in range(1,Nx+1)])
    diag_nNx = np.delete(diag_nNx, [i for i in range(Nx)])
        
    return diag, diag_p1, diag_pNx, diag_n1, diag_nNx,Lx,Ly
        
      
# Constants
hbar = 1
m0 = 1
e = 1
eps0 = 1/(4*math.pi)

# GaAs      

eps = 12.91*eps0               # 12.91 at low freq, 10.89 is high freq
me = 0.067*m0
mh = 0.51*m0                   # heavy holes of GaAs
mu = me*mh/(me+mh) 

# Black P
mu = 0.15*m0     
 
eps1 = 10*eps0
eps2 = 10*eps0
D = 10 /0.529177          # Width of layer input in Ang (->Bohr radii)
rho0 = D*eps/(eps1+eps2)    
# Black P
rho0 = 10.79 /0.529177
# Grid spacing values used in Peeters 2015: delta_min = 5e-5 Angstroms etc.

delta = 100
delta_min = 5 / 0.529177        # input in Angstroms
delta_max = 80 / 0.529177       # --> Bohr radii

# Create exponentially scaled grid
x,y = exp_grid(Nx,Ny,delta,delta_min,delta_max)
[XX, YY] = np.meshgrid(x, y)

# Calculate r values and get potential

r = (x**2 + y**2)**(1/2)
RR = (XX**2 + YY**2)**(1/2)

#V = Coulomb(RR) 
V = Keldysh(RR,eps1,eps2,rho0) 

# Get finite difference coefficients
diag, diag_p1, diag_pNx, diag_n1, diag_nNx,Lx,Ly = finite_differences(Nx,Ny,x,y)

# Construct Hamiltonian
            
H = sparse.diags(np.ravel(V))
H += (-hbar**2/(2*mu)) *sparse.diags(diag)          
H += (-hbar**2/(2*mu)) *sparse.diags(diag_p1, 1)  
H += (-hbar**2/(2*mu)) *sparse.diags(diag_pNx, Nx) 
H += (-hbar**2/(2*mu)) *sparse.diags(diag_n1, -1) 
H += (-hbar**2/(2*mu)) *sparse.diags(diag_nNx, -Nx)                 

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


####### Find lowest k eigenvalues ######

k=4

energies,states = eigsh(H, k=k, which='SA')
#energies,states = np.linalg.eig(H)

# Order
states = np.array([x for _, x in sorted(zip(energies, states.T), key=lambda pair: pair[0])])
energies = np.sort(energies)

# Hartree AU to meV
energies = energies*27.2114 *1000

print(energies)

# Prob. density
densities = (np.absolute(states))**2  #*4*math.pi*np.ravel(RR) #for radial prob density I think?

#for i in range(N):    
 #   print(densities[1][i])
 
# Choose starting eigenstate (will plot i, i+1, i+2, i+3)
i = 0

# Back to Angstroms for plotting
r = r  *0.529177
XX = XX*0.529177
YY = YY*0.529177
x = x  *0.529177
y = y  *0.529177

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
ax.set_xlabel("x (Ang)")
ax.set_ylabel("Wavefunction")
ax.set_title("n = 1")
#ax.set_ylim(-0.01,+0.1)
ax.text(xtext,ytext,("E = " +str(round(energy0,4))+ " eV"))

ax2 = fig1.add_subplot(2,2,2)
ax2.plot(x,densities[i+1].reshape((Nx,Ny))[int(Nx/2)],'bo',markersize=2)
ax2.set_xlabel("x (Ang)")
ax2.set_ylabel("Wavefunction")
ax2.set_title("n = 2")
#ax2.set_ylim(-0.01,+0.1)
ax2.text(xtext,ytext,("E = " +str(round(energy1,4))+ " eV"))

ax3 = fig1.add_subplot(2,2,3)
ax3.plot(x,densities[i+2].reshape((Nx,Ny))[int(Nx/2)],'bo',markersize=2)
ax3.set_xlabel("x (Ang)")
ax3.set_ylabel("Wavefunction")
ax3.set_title("n = 3")
#ax3.set_ylim(-0.01,+0.1)
ax3.text(xtext,ytext,("E = " +str(round(energy2, 4))+ " eV"))

ax4 = fig1.add_subplot(2,2,4)
ax4.plot(x,densities[i+3].reshape((Nx,Ny))[int(Nx/2)],'bo',markersize=2)
ax4.set_xlabel("x (Ang)")
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

density = densities[k-1].reshape((Nx,Ny))

fig3 = plt.figure()
ax = fig3.add_subplot(1,1,1)
c = ax.pcolormesh(XX,YY, density.T, shading='auto')
ax.set_title("(3,2,0)?")
ax.set_xlabel("x (Ang.)")
ax.set_ylabel("y (Ang.)")
fig3.colorbar(c,ax=ax)
fig3.tight_layout()
plt.show()





"""
            # Peeters
            diag[l] += -1/(hy[i+1]*(hy[i+1]+hy[i])/2)
            diag[l] += -1/(hy[i]  *(hy[i+1]+hy[i])/2)
            
            off_diag_p1[l] =  1/(hy[i+1]*(hy[i+1]+hy[i])/2)
            off_diag_n1[l] =  1/(hy[i]  *(hy[i+1]+hy[i])/2) 
            off_diag_pNy[l] = 1/(hx[j+1]*(hx[j+1]+hx[j])/2)
            off_diag_nNy[l] = 1/(hx[j]  *(hx[j+1]+hx[j])/2)
"""
