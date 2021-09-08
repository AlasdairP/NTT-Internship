"""
Finite differences for non-uniform 2D mesh
"""

import numpy as np

def second_derivative(Nx,Ny,x,y):
    
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
            
            # Boundary conditions
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
                
            # Majority of points
            else: 
            
                Lx[l] = (hx[i+1]+hx[i])/2
                Ly[l] = (hy[j+1]+hy[j])/2
                
                diag[l] =  -2/(hx[i]*hx[i+1])      
                diag[l] += -2/(hy[j]*hy[j+1])      
                
                diag_p1[l] =  1/(hx[i+1]*Lx[l])   
                diag_n1[l] =  1/(hx[i]  *Lx[l])   
                diag_pNx[l] = 1/(hy[j+1]*Ly[l])   
                diag_nNx[l] = 1/(hy[j]  *Ly[l])  
    
    # Delete final/first element (row) of the x(y) FDs to fit into H
    diag_p1 = np.delete(diag_p1, [N-1])
    diag_n1 = np.delete(diag_n1, [0])
    
    diag_pNx = np.delete(diag_pNx, [N - i for i in range(1,Nx+1)])
    diag_nNx = np.delete(diag_nNx, [i for i in range(Nx)])
        
    return diag, diag_p1, diag_pNx, diag_n1, diag_nNx, Lx, Ly 
    
    
def first_derivative_x(Nx,Ny,x):
    
    hx = np.zeros(Nx)
    
    for i in range(1,Nx):  
        hx[i] = x[i] - x[i-1]
    
    # First one is linearly spaced ...negligible error
    hx[0] = hx[1] + (hx[1] - hx[2]) 
    
    N = Nx*Ny
    
    diag = np.zeros(N)
    diag_p1 = np.zeros(N)
    diag_n1 = np.zeros(N) 
    
    for j in range(Ny):
        for i in range(Nx):
        
            l = j*Nx + i
            
            # Boundary conditions
            if i == (Nx-1):

                diag[l] = diag[l-1]
                diag_p1[l] = diag_p1[l-1]   
                diag_n1[l] = diag_n1[l-1]    

            elif j == (Ny-1):

                diag[l] = diag[l-Nx]
                diag_p1[l] = diag_p1[l-Nx]   
                diag_n1[l] = diag_n1[l-Nx]    
               
                
            # Majority of points
            else: 
                
                diag[l] = (hx[i+1] - hx[i])/(hx[i]*hx[i+1])           
                
                diag_p1[l] =  hx[i]/(hx[i+1] * (hx[i] + hx[i+1]))  
                diag_n1[l] = -hx[i+1]/(hx[i] * (hx[i] + hx[i+1]))    
    
    # Delete final/first element (row) of the x(y) FDs to fit into H
    # Don't need these here now as I have to keep the arrays full length to multiply by YY
    
    #diag_p1 = np.delete(diag_p1, [N-1])
    #diag_n1 = np.delete(diag_n1, [0])
        
    return diag, diag_p1, diag_n1
    
    
def first_derivative_y(Nx,Ny,y):
    
    hy = np.zeros(Ny)
    
    for i in range(1,Nx):  
        hy[i] = y[i] - y[i-1]
    
    # First one is linearly spaced ...negligible error
    hy[0] = hy[1] + (hy[1] - hy[2]) 
    
    N = Nx*Ny
    
    diag = np.zeros(N)
    diag_pNx = np.zeros(N)
    diag_nNx = np.zeros(N) 
    
    for j in range(Ny):
        for i in range(Nx):
        
            l = j*Nx + i
            
            # Boundary conditions
            if i == (Nx-1):

                diag[l] = diag[l-1]
                diag_pNx[l] = diag_pNx[l-1]   
                diag_nNx[l] = diag_nNx[l-1]    

            elif j == (Ny-1):

                diag[l] = diag[l-Nx]
                diag_pNx[l] = diag_pNx[l-Nx]   
                diag_nNx[l] = diag_nNx[l-Nx]    
               
            # Majority of points
            else: 
                
                diag[l] = (hy[j+1] - hy[j])/(hy[j]*hy[j+1])           
                
                diag_pNx[l] =  hy[j]/(hy[j+1] * (hy[j] + hy[j+1]))  
                diag_nNx[l] = -hy[j+1]/(hy[j] * (hy[j] + hy[j+1]))    
    
    # Delete final/first element (row) of the x(y) FDs to fit into H
    # Don't need these here now as I have to keep the arrays full length to multiply by XX
    
    #diag_pNx = np.delete(diag_pNx, [N - i for i in range(1,Nx+1)])
    #diag_nNx = np.delete(diag_nNx, [i for i in range(Nx)])
        
    return diag, diag_pNx, diag_nNx
    