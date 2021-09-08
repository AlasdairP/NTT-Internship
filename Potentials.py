"""
Potentials
"""

import numpy as np
import math
from scipy import special

def Coulomb(r, eps, d):
    Coulomb = -1/(4*math.pi*eps*np.sqrt(r**2 + d**2))
    return Coulomb
    
    
def Keldysh(r,eps1,eps2,rho0,d):
     
    # Zeroth order Struve function
    H0 = special.struve(0,(np.sqrt(r**2 + d**2)/rho0))
    
    # Neumann function
    Y0 = special.y0(np.sqrt(r**2 + d**2)/rho0)
    
    # Keldysh
    prefactor = -1*math.pi/((eps1+eps2)*rho0)
    # Original Peeters paper had 2*pi, but I think this is wrong
    
    #prefactor = -math.pi/(2*rho0)  # this would be true in vacuum surroundings
    
    V = prefactor*(H0 - Y0) 
    
    return V
    
    
    
    
"""
# Complex absorbing potential - a la Gawlas 2019, to describe tunnel ionisation
W = RR**6
# scaling param - specific to eigenstate and E field strength
eta = 1e38   
H_CAP = 1j * eta * W
H += sparse.diags(np.ravel(H_CAP))
"""