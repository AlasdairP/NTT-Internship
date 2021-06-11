 # -*- coding: utf-8 -*-
"""
Week4, w.b. Wed 9/06
Aim: working code for simplest exciton model, using Thibault's recommended Wannier/Keldysh method
    Fix the units problem
    
This uses a variable mesh in xy plane, and a Keldysh potential V(r)
Also uses relative coordinates x(y) = x(y)_e - x(y)_h
 
Hartree atomic units - Bohr radii, Hartree energy (= 1/2 Rydberg energy?)
Using the Hamiltonian from eq(4) of Peeters 2015 (they use Ry as energy units - factor of 2 different to Hartree - could cause problems)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy import constants

# Keldysh potential

def V(r,eps1,eps2,rho0):
     
    # Zeroth order Struve function
    H0 = scipy.special.struve(0,(r/rho0))
    
    # Neumann function
    Y0 = scipy.special.y0(r/rho0)
    
    # Keldysh
    prefactor = -2*math.pi/((eps1+eps2)*rho0)
    V = prefactor*(H0 - Y0)
        
    return V

def exp_grid(N,delta,delta_min,delta_max):

    # Create non-uniform mesh of points (x,y)
    x = np.zeros(N)
    y = np.zeros(N)
    
    for i in range(int(N/2),N):
        
        # Separation as per Peeters 2015
        h_i = delta_min + delta_max*(1-math.exp(-(i - N/2 -1)/delta)) # Removed -1 to stop weird behaviour around x(y)=0
        x[i] = x[i-1] + h_i                            # Not sure why it was weird with the -1 [added back in]
        y[i] = y[i-1] + h_i
        
        # Reflect to negative x,y values: [N/2,N] -> [0,N/2]
        x[N-i-1] = -x[i]
        y[N-i-1] = -y[i]
        
    return x,y
    
def finite_differences(x,y,N,mu):
    """
    Finite differences coefficients
    
    Dealing with first and last rows separately because you need
    two intervals, dx1 and dx2, to compute finite differences
    so first and last rows are tricky. 
    
    """
    ax = np.zeros(N)
    bx = np.zeros(N)
    cx = np.zeros(N)
    ay = np.zeros(N)
    by = np.zeros(N)
    cy = np.zeros(N)

    for i in range(1,N-1):   
        
        dx1 = x[i] - x[i-1]
        dx2 = x[i+1] - x[i]
        dy1 = y[i] - y[i-1]
        dy2 = y[i+1] - y[i]
        
        bx[i] = 2/(mu*dx1*dx2)  
        cx[i] = -2/(mu*dx2*(dx1+dx2))  
        ax[i] = -2/(mu*dx1*(dx1+dx2))

        by[i] = 2/(mu*dy1*dy2)
        cy[i] = -2/(mu*dy2*(dy1+dy2))  
        ay[i] = -2/(mu*dy1*(dy1+dy2))
        
    # First and last rows
    
    bx[0] = bx[1]   # Set b,c equal to 2nd row values (negligible error?)
    cx[0] = cx[1]  
    bx[N-1] = bx[N-2]
    ax[N-1] = bx[N-2] # Last row, set to previous value
    
    # Delete first(final) entry of a(c). They are zero (and need to have length N-1)
    ax = np.delete(ax,[0])   
    cx = np.delete(cx,[N-1]) 
    
    # Copy for y
    by[0] = by[1] 
    cy[0] = cy[1]
    by[N-1] = by[N-2]
    ay[N-1] = ay[N-2]
    ay = np.delete(ay,[0])
    cy = np.delete(cy,[N-1]) 
    
    return ax,bx,cx,ay,by,cy


############## Main #################
 
# Number of points, Nx = Ny = N
N = int(1000)

# Physical constants

hbar = 1  # Atomic units
m0 = 1    
me = 0.1  # 0.067*m0 for GaAs
mh = 0.5  # 0.45*m0
mu = 0.16 #me*mh/(me+mh)  

hbar = constants.hbar
eps0 = 1/(4*math.pi) # au 
eps1 = 1*eps0       # Barriers  
eps2 = 1*eps0
eps = 12*eps0      # Dielectric constant of layer
D = 20           # Width of layer in Bohr radii (atomic units of Bohr radii, not the 'Bohr radius' of the excitons in the TMD)
rho0 = D*eps/(eps1+eps2)    # Peeters 2015: rho0 = no. of layers*10.42 Angstroms (for P on SiO2)

# Grid spacing values used in Peeters 2015: delta_min = 5e-5 Angstroms etc.

delta = 100
delta_min = 5e-5*(1/0.529177)   # Bohr radii (1 Bohr radius = 0.53 Angstroms)
delta_max = 0.4 *(1/0.529177)

# Create exponentially scaled grid
x,y = exp_grid(N,delta,delta_min,delta_max)

# Get finite differences coefficients (just kinetic term, not including potential)
ax,bx,cx,ay,by,cy = finite_differences(x,y,N,mu)

# Calculate r values at each (x,y) point and get potential
r = (x**2 + y**2)**(1/2)
V = V(r,eps1,eps2,rho0)
    
"""
# Construct Hamiltonian, including the potential on the diagonals

H = [(bx0+by0+V0) (cx0+cy0)     0         0                 ...                       0
     (ax1+ay1)    (bx1+by1+V1)  (cx1+cy1) 0                 ...                       0
     0             ax2...       bx2...   cx2...             ...                       0
     .             .            .        .      ...
     .             .            .        .         (ax(N-1)+ay(N-1)) (bx(N-1)+by(N-1)+V(N-1) (cx(N-1)+cy(N-1))
     0             .            .        .          0                (ax(N)+ay(N))          (bx(N)+by(N)+V(N))    ]
"""
H = np.diag(bx) + np.diag(ax,k=-1) + np.diag(cx,k=1)
H += np.diag(by) + np.diag(ay,k=-1) + np.diag(cy,k=1)
H += np.diag(V)

# I now recover the infinite well solutions if I remove the V term


# Find eigenvalues of H
energies,states = np.linalg.eig(H)

# Convert Hartree atomic units to meV
energies = energies/27.2 *1000   

# New sorting method
states = np.array([x for _, x in sorted(zip(energies, states.T), key=lambda pair: pair[0])])
energies = np.sort(energies)

densities = (np.absolute(states))**2 # *4*math.pi*r for radial prob density I think?

# The first state (or two) has crazy large negative energies, so have filtered these out.
# Choose starting state
i=1

energy0 = energies[i]
energy1 = energies[i+1]
energy2 = energies[i+2]
energy3 = energies[i+3]

#print(np.isreal(states))           # The states are all real - unexpected?

# Plot first 4 states

fig1 = plt.figure()
ax = fig1.add_subplot(2,2,1)
ax.plot(x,densities[i],'bo',markersize=2,label='wavefunction')
ax.set_xlabel("x (a_0)")
ax.set_ylabel("Prob. density")
ax.set_title("n = 1")
#ax.set_xlim(-5,+5)
ax.text(0,0.005,("E = " +str(np.round(energy0,1))+ " meV"))
#ax.text(-200,0.07,("D = " + str(D) + " a_0 = " + str(round((D*0.53),1)) + "Ang."))

ax2 = fig1.add_subplot(2,2,2)
ax2.plot(x,densities[i+1],'bo',markersize=2)
ax2.set_xlabel("x (a_0)")
ax2.set_ylabel("Prob. density")
ax2.set_title("n = 2")
#ax2.set_xlim(-20,+20)
ax2.text(0,0.005,("E = " +str(np.round(energy1,1))+ " meV"))

ax3 = fig1.add_subplot(2,2,3)
ax3.plot(x,densities[i+2],'bo',markersize=2)
ax3.set_xlabel("x (a_0)")
ax3.set_ylabel("Prob. density")
ax3.set_title("n = 3")
#ax3.set_xlim(-50,+50)
ax3.text(0,0.005,("E = " +str(np.round(energy2, 1))+ " meV"))

ax4 = fig1.add_subplot(2,2,4)
ax4.plot(x,densities[i+3],'bo',markersize=2)
ax4.set_xlabel("x (a_0)")
ax4.set_ylabel("Prob. density")
ax4.set_title("n = 4")
#ax4.set_xlim(-100,+100)
ax4.text(0,0.005,("E = " +str(np.round(energy3,1))+ " meV"))
plt.tight_layout()

plt.show()

# Plot first three states on one plot

fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)
ax.plot(x,states[i],color='blue',label = str(np.round(energy0,1))+' meV')
ax.plot(x,states[i+1],color='green',label = str(np.round(energy1,1))+' meV')
ax.plot(x,states[i+2],color='red',label = str(np.round(energy2,1))+' meV')
#ax.set_xlim(-10,+20)
ax.set_ylabel("Wavefunction")
ax.set_xlabel("x (a_0)")
ax.set_title("First 3 states")
plt.legend()
plt.show()

# Plot the energies as a function of n

fig3 = plt.figure()
ax = fig3.add_subplot(1,1,1)
ax.plot(np.arange(0,20),energies[0:20], 'bo')
ax.set_title("Energy vs n")
ax.set_xlabel("State index n")
ax.set_ylabel("Energy (meV)")
ax.set_ylim(-500, 50)
plt.show()

"""
# Plot energies of states against state index

number_of_states = 950
state_indices = np.arange(number_of_states)
energy_list = np.zeros(number_of_states)
for i in range(number_of_states):
    energy_list[i] = energies[i]

fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(state_indices,energy_list)
ax5.set_xlabel("State index")
ax5.set_ylabel("Energy (meV)")
ax5.set_ylim(-20000,+20000)
ax5.set_title("Energy eigenvalues")
ax5.text(300,-500,("D = " + str(D) + " a_0 = " + str(round((D*0.53),1)) + "Angstroms"))
plt.show()

"""
"""
# Plot potential

V_plot = np.zeros(N)
for i in range(N):
    V_plot[i] = V(r[i],eps1,eps2,rho0)
    
ax.plot(x,V_plot,'ro',markersize=2,label='potential')
ax.legend()
"""




