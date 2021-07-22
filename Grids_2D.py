"""
Exponential and uniform grid in 2D
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def exp_grid(Nx, Ny, delta,delta_min,delta_max):

    # Create non-uniform mesh of points (x,y)
    x = np.zeros(Nx)
    y = np.zeros(Ny)
    
    for i in range(int(Nx/2),Nx):
        
        # Separation as per Peeters 
        hx_i = delta_min + delta_max*(1-math.exp(-(i - Nx/2)/delta)) # removed -1 
        x[i] = x[i-1] + hx_i                            
        
        # Reflect to negative x: [N/2,N] -> [0,N/2]
        x[Nx-i-1] = -x[i]
    
    # Copy for y    
    for i in range(int(Ny/2),Ny):
        hy_i = delta_min + delta_max*(1-math.exp(-(i - Ny/2)/delta))                            
        y[i] = y[i-1] + hy_i
        y[Ny-i-1] = -y[i]
        
    # Shift all points towards origin to correct double space at origin
    
    for i in range(int(Nx/2),Nx):   # positive x
        x[i] = x[i] - delta_min/2
    for i in range(0,int(Nx/2)):    # negative x
        x[i] = x[i] + delta_min/2
        
    for i in range(int(Ny/2),Ny):   # positive y
        y[i] = y[i] - delta_min/2
    for i in range(0,int(Ny/2)):    # negative y
        y[i] = y[i] + delta_min/2
      
    #L = x[Nx-1] - x[0]
    #x_uniform = np.linspace(-L/2, L/2, Nx)
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot(x,x_uniform,'o')
    #plt.show()

    return x,y
    
def uniform_grid(Nx,Ny,L):
    
    hx = L/Nx
    hy = L/Ny
    
    x = np.linspace(-L/2, L/2-hx, Nx)
    y = np.linspace(-L/2, L/2-hy, Ny)
    
    return x,y