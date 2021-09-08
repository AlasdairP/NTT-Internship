"""
Master. 
Choose uniform or exponential 2D grid.
Choose (screened) Coulomb or Keldysh potential.
Main calculations done in Hatree AU.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import constants
from scipy import sparse
from scipy import optimize
from scipy.sparse.linalg import eigs,eigsh

# My files
import Grids_2D
import Finite_differences
import Potentials
import Symmetrise_and_solve
import Plotting

# Constants (AU)
hbar = 1
m0 = 1
e = 1
eps0 = 1/(4*math.pi)
Bohr_magneton = 1/2

# Unit conversions
Bohr_to_Ang = constants.value('Bohr radius') *1e10         # 0.529
Hartree_in_eV = constants.value('Hartree energy in eV')     # 27.2
kV_per_cm_to_AU = constants.value('Hartree energy')/(constants.e*constants.value('Bohr radius')) *1e-5   # 5.14e6
Tesla_to_AU = constants.hbar/(constants.e*constants.value('Bohr radius')**2)   # 2.35e5
vel_SI_to_AU = constants.value('Bohr radius')*constants.value('Hartree energy')/constants.hbar   # 2.19e6

######################## Main ###############################

# Get all the system params and variables from INPUT file.
variables_dict = {}
with open("INPUT.txt") as f:
    for line in f:
        variable_name = line.partition('=')[0]
        variable_name = variable_name.strip()
        value = line.partition('=')[2]
        value = value.strip()
        variables_dict[variable_name] = value
        
# Number of points
Nx = int(variables_dict["Nx"])
Ny = int(variables_dict["Ny"])
N = Nx*Ny

Electric = True if variables_dict["electric"] == 'True' else False
Magnetic = True if variables_dict["magnetic"] == 'True' else False
plot_potential = True if variables_dict["plot_potential_every_iteration"] == 'True' else False
plot_PD = True if variables_dict["plot_PD_every_iteration"] == 'True' else False
plot_phase = True if variables_dict["plot_phase_every_iteration"] == 'True' else False
    
######################## Get material/system params ##########
eps = float(variables_dict["eps"]) * eps0 
eps1 = float(variables_dict["eps1"]) * eps0  
eps2 = float(variables_dict["eps2"]) * eps0 
me = float(variables_dict["me"]) * m0
mh = float(variables_dict["mh"]) * m0
M = me + mh
mu = me*mh/M   
nu = (mh*me)/(mh - me)
rho0 = float(variables_dict["rho0"]) / Bohr_to_Ang   # Divide for Ang to Bohr
#D = 5 / Bohr_to_Ang        
#rho0 = D*eps/(eps1+eps2)

spin = float(variables_dict["spin"])              # +1 or -1 or 0 if not including spin
g_factor = float(variables_dict["g_factor"])

######################### Create grid #######################

if variables_dict["grid"] == "exp":
    grid = "Exp. grid"
    delta = float(variables_dict["delta"])
    delta_min = float(variables_dict["delta_min"]) / Bohr_to_Ang        
    delta_max = float(variables_dict["delta_max"]) / Bohr_to_Ang         
    x,y,hx,hy = Grids_2D.exp_grid(Nx,Ny,delta,delta_min,delta_max)
    
elif variables_dict["grid"] == "uniform":
    grid = "Uniform grid"
    L = float(variables_dict["L"]) / Bohr_to_Ang
    x,y = Grids_2D.uniform_grid(Nx,Ny,L)
    
elif variables_dict["grid"] == "stepped":
    grid = "Stepped grid"
    sep1 = float(variables_dict["sep1"]) / Bohr_to_Ang        
    sep2 = float(variables_dict["sep2"]) / Bohr_to_Ang
    N1 = int(variables_dict["N1"])        
    N2 = int(variables_dict["N2"]) 
    x,y,hx,hy = Grids_2D.stepped_grid(Nx,Ny,sep1,sep2,N1,N2)

[XX, YY] = np.meshgrid(x, y)

# For plotting
XX_Ang = XX*Bohr_to_Ang
YY_Ang = YY*Bohr_to_Ang
x_Ang = x*Bohr_to_Ang
y_Ang = y*Bohr_to_Ang

########## Get FDs for 2nd derivative in Schrodinger Eq, d^2/dx^2 + d^2/dy^2 ###########

diag, diag_p1, diag_pNx, diag_n1, diag_nNx, Lx, Ly = Finite_differences.second_derivative(Nx,Ny,x,y)

############################## Get potential ##################

# Choose soft-core d value in Angstroms
d = float(variables_dict["soft_core"])

# Calculate r values 
RR = (XX**2 + YY**2)**(1/2)

if variables_dict["potential"] == "Coulomb":
    V = Potentials.Coulomb(RR,eps,d)
if variables_dict["potential"] == "Keldysh":
    V = Potentials.Keldysh(RR,eps1,eps2,rho0,d)
    # fudge factor - still not 100% sure on this but it gives the right numbers
    V = V / Hartree_in_eV
    
############################ Build basic Hamiltonian H_0 ##############

H = sparse.diags(np.ravel(V))
H += (-hbar**2/(2*mu)) *sparse.diags(diag)          
H += (-hbar**2/(2*mu)) *sparse.diags(diag_p1, 1)  
H += (-hbar**2/(2*mu)) *sparse.diags(diag_pNx, Nx) 
H += (-hbar**2/(2*mu)) *sparse.diags(diag_n1, -1) 
H += (-hbar**2/(2*mu)) *sparse.diags(diag_nNx, -Nx)
    
############################ Add external E or B field and Solve SE #############

# Choose number of eigenstates, runs slower for more states
k = int(variables_dict["eigenstates"])                             
                                            
if Electric:                   
    
    if Magnetic:               # Both
        
        # field strength range - input in Tesla
        B_min = float(variables_dict["B_min"])
        B_max = float(variables_dict["B_max"])
        number = int(variables_dict["B_number"])
        B_Tesla = np.linspace(B_min,B_max,number)
        B = B_Tesla / Tesla_to_AU                   # T -> AU
        
        # Start with constant E field
        F_kV = float(variables_dict["E_min"])
        F = F_kV / kV_per_cm_to_AU
        
        # Constant velocity
        vx = float(variables_dict["vx"]) / vel_SI_to_AU
        vy = float(variables_dict["vy"]) / vel_SI_to_AU
        
        # Same two options for B field as in the B field only section.
        diag, diag_n1, diag_p1 = Finite_differences.first_derivative_x(Nx,Ny,x)
            
        energy_lists = [np.zeros(number) for j in range(k)]
        for i in range(number):
                
            B_term1 = (1/(2*mu)) * e**2 * B[i]**2 * YY**2
            
            B_term2_pref = np.ravel((hbar/(nu)) * 1j * e * B[i] * -YY)
            B_diag = B_term2_pref * diag
            # if x derivative
            B_diag_n1 = B_term2_pref * diag_n1
            B_diag_p1 = B_term2_pref * diag_p1
            # if y derivative
            #B_diag_nNx = B_term2_pref * diag_nNx
            #B_diag_pNx = B_term2_pref * diag_pNx
            # if x derivative
            B_diag_p1_del = np.delete(B_diag_p1, [N-1])
            B_diag_n1_del = np.delete(B_diag_n1, [0])
            # if y derivative
            #B_diag_pNx_del = np.delete(B_diag_pNx, [N - i for i in range(1,Nx+1)])
            #B_diag_nNx_del = np.delete(B_diag_nNx, [i for i in range(Nx)])
            
            # TRUE ELECTRIC
            E = - e * F * XX
            
            # EFFECTIVE ELECTRIC (VELOCITY DEPENDENT)
            E_eff = -e * B[i] * (vy*XX - vx*YY)
            
            # ZEEMAN
            Zeeman = spin * g_factor * Bohr_magneton * B[i] * np.ones(Nx*Ny)
            
            # CoM Kinetic energy (classical) - this doesn't really add much insight I don't think
            #CoM_kinetic = 1/2 * M * (vx**2 + vy**2) * np.ones(Nx*Ny)
            
            # Plot linecut of potential
            if plot_potential:
                V_plot =  V + E + B_term1 + E_eff + (B_diag + B_diag_n1 + B_diag_p1).reshape((Ny,Nx))
                fig = Plotting.Potential_linecut(x_Ang, V_plot, y_cut=70)
                plt.show()
                
            # Construct Hamiltonian
            
            #H += sparse.diags(CoM_kinetic)
            H += sparse.diags(np.ravel(B_term1))
            H += sparse.diags(B_diag)
            H += sparse.diags(B_diag_n1_del, -1)  
            H += sparse.diags(B_diag_p1_del, +1)
            H += sparse.diags(np.ravel(E_eff))
            H += sparse.diags(Zeeman) 
            H += sparse.diags(np.ravel(E))

            # Solve (energies returned in meV)
            energies,states = Symmetrise_and_solve.solve(H,Lx,Ly,k)
        
            # Add energy of each state to the list (for field iteration i)
            for state_index in range(k):
                energy_lists[state_index][i] = energies[state_index]
            
            print(np.real_if_close(energies,tol=1000))
            
            # Remove ready for next iteration
            #H -= sparse.diags(CoM_kinetic)
            H -= sparse.diags(np.ravel(B_term1))
            H -= sparse.diags(B_diag)
            H -= sparse.diags(B_diag_n1_del, -1)  
            H -= sparse.diags(B_diag_p1_del, +1)
            H -= sparse.diags(np.ravel(E_eff))
            H -= sparse.diags(Zeeman)
            H -= sparse.diags(np.ravel(E))
            
            # Plot probability density and/or phase, choose state_index (< k)
            if plot_PD:
                state_index = int(input("Choose state index to plot (PD) (< # eigenstates): ")) 
                fig = Plotting.Probability_density(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid)
                plt.show()
            if plot_phase:
                state_index = int(input("Choose state index to plot (Phase) (< # eigenstates): "))
                fig = Plotting.Phase(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid)
                plt.show()
                
        # Plot binding energy vs field
        fig = Plotting.Energy_vs_field(energy_lists,k,Electric,Magnetic,F_kV,B_Tesla,grid)
        plt.show()
            
    else:                         # Electric only
        
        # Range - input in KV/cm
        F_min = float(variables_dict["E_min"])
        F_max = float(variables_dict["E_max"])
        number = int(variables_dict["E_number"])
        F_kV = np.linspace(F_min, F_max, number)
        F = F_kV / kV_per_cm_to_AU  # -> AU
        B_Tesla = 0
        
        # Create lists to store energies for each state
        energy_lists = [np.zeros(number) for j in range(k)]
    
        for i, F_strength_i in enumerate(F):
            
            E = - e * F_strength_i * XX
            H += sparse.diags(np.ravel(E))

            # Plot linecut of potential if requested
            if plot_potential:
                V_plot =  V + E.reshape((Ny,Nx))
                fig = Plotting.Potential_linecut(x_Ang, V_plot, y_cut=70)
                plt.show()
            
            # Solve (energies returned in meV)
            energies,states = Symmetrise_and_solve.solve(H,Lx,Ly,k)
            
            # Add energy of each state to the list (for field iteration i)
            for state in range(k):
                energy_lists[state][i] = np.real_if_close(energies[state],tol=100)
            
            print(np.real_if_close(energies,tol=100))
            
            # Remove this E field from H ready for the next one
            H -= sparse.diags(np.ravel(E))
            
            # Plot probability density and phase if requested, choose state_index (< k)
            if plot_PD:
                state_index = int(input("Choose state index to plot (< # eigenstates): "))
                fig = Plotting.Probability_density(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid)
                plt.show()
            if plot_phase:
                state_index = int(input("Choose state index to plot (Phase) (< # eigenstates): "))
                fig = Plotting.Phase(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid)
                plt.show()
            
        # Plot binding energy vs field
        fig = Plotting.Energy_vs_field(energy_lists,k,Electric,Magnetic,F_kV,B_Tesla,grid)
        plt.show()

elif Magnetic:               # Only
    
    # field strength - input in Tesla
    B_min = float(variables_dict["B_min"])
    B_max = float(variables_dict["B_max"])
    number = int(variables_dict["B_number"])
    B_Tesla = np.linspace(B_min,B_max,number)
    B = B_Tesla / Tesla_to_AU  # -> AU
    F_kV = np.zeros(number)

    vx = float(variables_dict["vx"]) / vel_SI_to_AU
    vy = float(variables_dict["vy"]) / vel_SI_to_AU
    
    energy_lists = [np.zeros(number) for j in range(k)]

    """
    2 options currently, both for B in +z direction:
    1. A = (0   , B.x , 0)   [Landau gauge]
    Use XX in first two lines, y derivative, all diags "pNx", "nNx" 
    2. A = (-B.y,  0  , 0)
    Use -YY in first two lines, x derivative, all diaga "p1","n1" 
    note: symmetric gauge would be half of each of these I think.
    """

    for i, B_i in enumerate(B):
        
        # ZEEMAN
        Zeeman = spin * g_factor * Bohr_magneton * B_i * np.ones(Nx*Ny)
        
        # VELOCITY DEPENDENT EFFECTIVE ELECTRIC FIELD - should be linear (but currently looks quadratic)
        E_eff = -e * B_i * (vy*XX - vx*YY)  
        
        # MAIN B FIELD TERMS
        B_term1 = (1/(2*mu)) * e**2 * B_i**2 * XX**2
        B_term2_pref = np.ravel((hbar/(nu)) * 1j * e * B_i * XX)
        
        diag, diag_nNx, diag_pNx = Finite_differences.first_derivative_y(Nx,Ny,y)
        
        B_diag = B_term2_pref * diag
        # if x derivative
        #B_diag_n1 = B_term2_pref * diag_n1
        #B_diag_p1 = B_term2_pref * diag_p1
        # if y derivative
        B_diag_nNx = B_term2_pref * diag_nNx
        B_diag_pNx = B_term2_pref * diag_pNx
        
        # Plot linecut of potential
        if plot_potential:
            V_plot =  V + B_term1 + E_eff + (B_diag + B_diag_nNx + B_diag_pNx).reshape((Ny,Nx))
            fig = Plotting.Potential_linecut(x_Ang, V_plot, y_cut=70)
            plt.show() 
        
        # if x derivative
        #B_diag_p1 = np.delete(B_diag_p1, [N-1])
        #B_diag_n1 = np.delete(B_diag_n1, [0])
        # if y derivative
        B_diag_pNx = np.delete(B_diag_pNx, [N - i for i in range(1,Nx+1)])
        B_diag_nNx = np.delete(B_diag_nNx, [i for i in range(Nx)])
            
        # Construct Hamiltonian
        
        H += sparse.diags(np.ravel(B_term1))
        H += sparse.diags(B_diag)
        H += sparse.diags(B_diag_nNx, -Nx)  
        H += sparse.diags(B_diag_pNx, +Nx)
        H += sparse.diags(Zeeman) 
        H += sparse.diags(np.ravel(E_eff))
        
        # Solve (energies returned in meV)
        energies,states = Symmetrise_and_solve.solve(H,Lx,Ly,k)
        
        # Add energy of each state to the list (for field iteration i)
        for state in range(k):
            energy_lists[state][i] = energies[state]
        
        print(np.real_if_close(energies,tol=100))
        
        # Remove ready for next iteration
        H -= sparse.diags(np.ravel(B_term1))
        H -= sparse.diags(B_diag)
        H -= sparse.diags(B_diag_nNx, -Nx)  
        H -= sparse.diags(B_diag_pNx, +Nx)
        H -= sparse.diags(Zeeman)
        H -= sparse.diags(np.ravel(E_eff))
        
        # Plot probability density, choose state_index (< k)
        if plot_PD:
            state_index = int(input("Choose state index to plot (< # eigenstates): ")) 
            fig = Plotting.Probability_density(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid)
            plt.show()
        if plot_phase:
            state_index = int(input("Choose state index to plot (Phase) (< # eigenstates): "))
            fig = Plotting.Phase(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid)
            plt.show()
        
    # Plot binding energy vs field
    fig = Plotting.Energy_vs_field(energy_lists,k,Electric,Magnetic,F_kV,B_Tesla,grid)
    plt.show()
    
else:     # No fields 
    
    F_kV = 0
    B_Tesla = 0
    # Solve    
    energies,states = Symmetrise_and_solve.solve(H,Lx,Ly,k)
    print("Binding energies: " + str(energies))
    # Plot probability density - choose state
    i=0
    state_index = int(input("Choose state index to plot (< # eigenstates): ")) 
    fig = Plotting.Probability_density(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid)
    plt.show()

#################### Curve fitting #######################

# Define function
def func(x,a,b,c):
    return a*x**2 + b*x + c
 
if Electric:
    if not Magnetic:
        # Find polarisability of ground state
        xdata = F_kV
        ydata = energy_lists[0]
        optimised_params, covariance = optimize.curve_fit(func, xdata, ydata)
        pol = optimised_params[0]
        #pol = pol*10000      # Conversion meV cm^2 (kV)^(-2) --> e Ang^2 / mV, literature sometimes gives it in either.
        print("2D polarisability of GS = " + str(pol) + " meV cm^2 (kV)^(-2)")
        
if Magnetic:
    xdata = B_Tesla
    ydata = energy_lists[0]
    optimised_params, covariance = optimize.curve_fit(func, xdata, ydata)
    sigma = optimised_params[0]
    Zeeman_factor = optimised_params[1]
    print("GS diamag. shift = " + str(sigma) + "meV / T^2")
    radius = math.sqrt(8*mu*constants.m_e* abs(sigma)/1000 * constants.e)/constants.e *1e10
    print("GS Radius from diamag. shift = " +str(radius) + "Ang")
    print("GS Linear (Zeeman + E_eff) shift = " +str(Zeeman_factor) + " meV/T")










