"""
Exponential and uniform grid in 2D
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def exp_grid(Nx,Ny,delta,delta_min,delta_max):

    # Create non-uniform mesh of points (x,y)
    x = np.zeros(Nx)
    y = np.zeros(Ny)
    hx = np.zeros(Nx)
    hy = np.zeros(Ny)
    for i in range(int(Nx/2),Nx):
        
        # Separation as per Peeters 
        hx[i] = delta_min + delta_max*(math.exp(+(i - Nx/2)/delta)) # removed -1, adding +1 or +2 helps the weird -> 0 at axes behaviour
        x[i] = x[i-1] + hx[i]
        
        # Reflect to negative x: [N/2,N] -> [0,N/2]
        if (Nx-i-1) == i:
            pass
        else:
            x[Nx-i-1] = -x[i]
            
    # Copy for y    
    for i in range(int(Ny/2),Ny):
        hy[i] = delta_min + delta_max*(math.exp(+(i - Ny/2)/delta))                            
        y[i] = y[i-1] + hy[i]
        if (Ny-i-1) == i:
            pass
        else:
            y[Ny-i-1] = -y[i]
    
    # Shift all points towards origin to correct double space at origin
    if Nx%2 == 0:
        #print("even Nx")
        for i in range(int(Nx/2),Nx):   # positive x
            x[i] = x[i] - hx[int(Nx/2)]/2
        for i in range(0,int(Nx/2)):    # negative x
            x[i] = x[i] + hx[int(Nx/2)]/2
    
    else:    
        #print("odd Nx")
        x[int(Nx/2)] = 0
        for i in range(int(Nx/2)+1,Nx):   # positive x
            x[i] = x[i] - hx[int(Nx/2)]
        for i in range(0,int(Nx/2)):    # negative x
            x[i] = x[i] + hx[int(Nx/2)]
    
    if Ny%2 == 0:     
        #print("even Ny")   
        for i in range(int(Ny/2),Ny):   # positive y
            y[i] = y[i] - hy[int(Ny/2)]/2
        for i in range(0,int(Ny/2)):    # negative y
            y[i] = y[i] + hy[int(Ny/2)]/2
    
    else:    
        #print("odd Ny")
        y[int(Ny/2)] = 0
            
        for i in range(int(Ny/2)+1,Ny):   # positive y
            y[i] = y[i] - hy[int(Ny/2)]
        for i in range(0,int(Ny/2)):    # negative y
            y[i] = y[i] + hy[int(Ny/2)]
    
    # Plot x (or y) against a uniform 'label' to see the exp scaling
    """  
    L = x[Nx-1] - x[0]
    x_uniform = np.linspace(-L/2, L/2, Nx)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,x_uniform,'o')
    plt.show()
    
    L = y[Ny-1] - y[0]
    y_uniform = np.linspace(-L/2, L/2, Ny)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(y,y_uniform,'o')
    plt.show()
    """
    # Keep positive half of hx,hy i.e. get rid of all the zeros
    hx = hx[int(Nx/2):]
    hy = hy[int(Ny/2):]
    
    # Plot separation - only works for even Nx currently.
    L = x[Nx-1] - x[0]
    x_uniform = np.linspace(0, L/2, int(Nx/2))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_uniform*0.529,hx*0.529,'o')
    ax.plot(x[int(Nx/2):]*0.529,hx*0.529,'o')
    ax.set_title("Separations")
    ax.set_ylabel("hx (Ang.)")
    ax.set_xlabel("x (orange), x_uniform (blue) (Ang.)")
    plt.show()
    
    x = x/0.529
    n = len (x [x < 30])
    print("fraction of 'useful' points (x,y < 30 Ang.) = " + str(n/len(x)))
    x = x*0.529
    
    return x,y,hx,hy
    
def uniform_grid(Nx,Ny,L):
    
    hx = L/Nx
    hy = L/Ny
    
    x = np.linspace(-L/2, L/2-hx, Nx)
    y = np.linspace(-L/2, L/2-hy, Ny)
    
    return x,y
    
def stepped_grid(Nx,Ny,sep1,sep2,N1,N2):
    
    step = (N1/2)*sep1
    L = step + (N2/2)*sep2
    x1 = np.linspace(-step,step,N1, endpoint=False)
    x2pos = np.linspace(step,L,int(N2/2))
    x2neg = np.linspace(-L,-step,int(N2/2), endpoint=False)
    x = np.concatenate((x2neg,x1,x2pos))
    
    hx1 = sep1*np.ones(int(N1/2))
    hx2 = sep2*np.ones(int(N2/2))
    hx = np.concatenate((hx1,hx2))
    
    step = (N1/2)*sep1
    L = step + (N2/2)*sep2
    y1 = np.linspace(-step,step,N1, endpoint=False)
    y2pos = np.linspace(step,L,int(N2/2))
    y2neg = np.linspace(-L,-step,int(N2/2), endpoint=False)
    y = np.concatenate((y2neg,y1,y2pos))
    
    hy1 = sep1*np.ones(int(N1/2))
    hy2 = sep2*np.ones(int(N2/2))
    hy = np.concatenate((hy1,hy2))
    
    
    L = x[Nx-1] - x[0]
    x_uniform = np.linspace(-L/2, L/2, Nx)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,x_uniform,'o')
    plt.show()
    
        # Plot separation - only works for even Nx currently.
    L = x[Nx-1] - x[0]
    x_uniform = np.linspace(0, L/2, int(Nx/2))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_uniform*0.529,hx*0.529,'o')
    ax.set_title("Separations")
    ax.set_ylabel("hx (Ang.)")
    ax.set_xlabel("x (Ang.)")
    plt.show()
    
    return x,y,hx,hy
    
    
    