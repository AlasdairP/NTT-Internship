import numpy as np
from scipy.sparse.linalg import eigs,eigsh
from scipy import constants
from scipy import sparse

Hartree_in_eV = constants.value('Hartree energy in eV')
# 27.2

def solve(H,Lx,Ly,k):
    
    # Symmetrise 
    
    Lx = np.sqrt(Lx)
    Ly = np.sqrt(Ly)
    L = Lx*Ly  
    L_inv = 1/L    
    L = sparse.diags(L)
    L_inv = sparse.diags(L_inv)
    H = L_inv*L*L*H*L_inv
    
    # Solve Schrodinger Eq: choose eigsh (real symmetric, complex Hermitian) or eigs
    
    #energies,states = eigsh(H, k=k, which='SA')
    energies,states = eigs(H, k=k, which='SR')
    
    # Order
    states = np.array([st for _, st in sorted(zip(energies, states.T), key=lambda pair: pair[0])])
    energies = np.sort(energies)
    
    # Hartree AU -> meV
    energies = energies*Hartree_in_eV *1000
    
    return energies,states